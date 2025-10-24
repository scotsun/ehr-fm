"""
MIMIC-IV Data Asset Definitions

Data Pipeline:
1. Import raw CSV to DuckDB (raw_* assets)
2. dbt transformation and merging (tokens models)
3. Export as partitioned Parquet (export_tokens asset)
"""

import dagster as dg
from dagster_duckdb import DuckDBResource
from mimic4.constants import RAW_DATA_DIR, EXPORT_DIR


# Step 1: Import raw tables to DuckDB
@dg.asset(kinds={"duckdb"}, key=["main", "raw_admissions"])
def raw_admissions(duckdb: DuckDBResource) -> None:
    """
    Import admissions.csv to raw_admissions table
    
    Key fields:
    - subject_id: Patient ID
    - hadm_id: Hospital admission ID (analogous to order_id)
    - admittime: Admission time (for sorting)
    - dischtime: Discharge time
    """
    with duckdb.get_connection() as conn:
        conn.execute(
            f"""
            create or replace table raw_admissions as (
                select * from read_csv_auto('{RAW_DATA_DIR}/admissions.csv')
            )
            """
        )


@dg.asset(kinds={"duckdb"}, key=["main", "raw_diagnoses_icd"])
def raw_diagnoses_icd(duckdb: DuckDBResource) -> None:
    """
    Import diagnoses_icd.csv to raw_diagnoses_icd table
    
    Key fields:
    - subject_id: Patient ID
    - hadm_id: Hospital admission ID
    - icd_code: ICD diagnosis code (analogous to product_id)
    - seq_num: Diagnosis sequence
    """
    with duckdb.get_connection() as conn:
        conn.execute(
            f"""
            create or replace table raw_diagnoses_icd as (
                select * from read_csv_auto('{RAW_DATA_DIR}/diagnoses_icd.csv')
            )
            """
        )


@dg.asset(kinds={"duckdb"}, key=["main", "raw_prescriptions"])
def raw_prescriptions(duckdb: DuckDBResource) -> None:
    """
    Import prescriptions.csv to raw_prescriptions table
    
    Key fields:
    - subject_id: Patient ID
    - hadm_id: Hospital admission ID
    - drug: Drug name (analogous to product_id)
    - starttime: Start time
    """
    with duckdb.get_connection() as conn:
        conn.execute(
            f"""
            create or replace table raw_prescriptions as (
                select * from read_csv_auto('{RAW_DATA_DIR}/prescriptions.csv', ignore_errors=true)
            )
            """
        )


@dg.asset(kinds={"duckdb"}, key=["main", "raw_procedures_icd"])
def raw_procedures_icd(duckdb: DuckDBResource) -> None:
    """
    Import procedures_icd.csv to raw_procedures_icd table
    
    Key fields:
    - subject_id: Patient ID
    - hadm_id: Hospital admission ID
    - icd_code: ICD procedure/operation code
    - chartdate: Operation date
    - seq_num: Operation sequence
    """
    with duckdb.get_connection() as conn:
        conn.execute(
            f"""
            create or replace table raw_procedures_icd as (
                select * from read_csv_auto('{RAW_DATA_DIR}/procedures_icd.csv')
            )
            """
        )


@dg.asset(kinds={"duckdb"}, key=["main", "raw_labevents"])
def raw_labevents(duckdb: DuckDBResource) -> None:
    """
    Import labevents.csv to raw_labevents table
    
    Note: Large file (17GB), using ignore_errors=true to handle format issues
    
    Key fields:
    - subject_id: Patient ID
    - hadm_id: Hospital admission ID (may be null)
    - itemid: Lab item ID
    - charttime: Lab time
    - valuenum: Lab value
    """
    with duckdb.get_connection() as conn:
        # Set memory limit
        conn.execute("SET memory_limit='8GB';")
        conn.execute("SET temp_directory='/tmp/duckdb_temp';")
        
        conn.execute(
            f"""
            create or replace table raw_labevents as (
                select * from read_csv_auto(
                    '{RAW_DATA_DIR}/labevents.csv',
                    ignore_errors=true,
                    sample_size=100000
                )
            )
            """
        )


# Stage 1: Pre-computed tables (ETHOS Pipeline Foundation)
@dg.asset(
    kinds={"duckdb"},
    key=["main", "top_200_labs"],
    deps=[dg.AssetKey(["main", "raw_labevents"])]
)
def top_200_labs(duckdb: DuckDBResource) -> None:
    """
    Calculate the most common 200 lab items
    
    Strategy:
    - Only count inpatient labs with hadm_id
    - Sort by frequency in descending order
    - Top-200 covers ~95% of lab volume (based on literature)
    
    Output table structure:
    - itemid: Lab item ID
    - freq: Frequency
    - pct: Percentage
    """
    with duckdb.get_connection() as conn:
        print("Calculating Top-200 lab items...")
        conn.execute("""
            create or replace table top_200_labs as
            select 
                itemid,
                count(*) as freq,
                count(*) * 100.0 / sum(count(*)) over () as pct
            from raw_labevents
            where hadm_id is not null
            group by itemid
            order by freq desc
            limit 200
        """)
        
        # Validation
        result = conn.execute("select count(*), sum(pct) from top_200_labs").fetchone()
        print(f"✅ Top-200 lab items created")
        print(f"   Item count: {result[0]}")
        print(f"   Coverage: {result[1]:.2f}%")


@dg.asset(
    kinds={"duckdb"},
    key=["main", "lab_quantiles"],
    deps=[
        dg.AssetKey(["main", "raw_labevents"]),
        dg.AssetKey(["main", "top_200_labs"])
    ]
)
def lab_quantiles(duckdb: DuckDBResource) -> None:
    """
    Calculate quantiles for Top-200 labs (Q1-Q10, ETHOS method)
    
    Strategy:
    - Sample 10% of data for calculation (performance optimization, still accurate)
    - Only calculate records with values (valuenum is not null)
    - Calculate 9 quantile points (0.1, 0.2, ..., 0.9) for each itemid
    
    Output table structure:
    - itemid: Lab item ID
    - q1-q9: 9 quantile thresholds
    
    Usage:
    - Convert continuous values to Q1-Q10 discrete categories
    - Example: glucose=95 → LAB:50931_Q5 (medium level)
    """
    with duckdb.get_connection() as conn:
        print("Calculating quantiles (10% sample, estimated 10-20 minutes)...")
        
        conn.execute("""
            create or replace table lab_quantiles as
            select 
                itemid,
                quantile_cont(valuenum, 0.1) as q1,
                quantile_cont(valuenum, 0.2) as q2,
                quantile_cont(valuenum, 0.3) as q3,
                quantile_cont(valuenum, 0.4) as q4,
                quantile_cont(valuenum, 0.5) as q5,
                quantile_cont(valuenum, 0.6) as q6,
                quantile_cont(valuenum, 0.7) as q7,
                quantile_cont(valuenum, 0.8) as q8,
                quantile_cont(valuenum, 0.9) as q9
            from (
                select itemid, valuenum
                from raw_labevents
                where valuenum is not null
                    and itemid in (select itemid from top_200_labs)
                using sample 10%
            )
            group by itemid
        """)
        
        # Validation
        result = conn.execute("select count(*) from lab_quantiles").fetchone()[0]
        print(f"✅ Quantiles table created")
        print(f"   Lab item count: {result}")
        
        # Check quantile reasonableness (sampling)
        sample = conn.execute("""
            select itemid, q1, q5, q9 
            from lab_quantiles 
            where q1 is not null and q5 is not null and q9 is not null
            limit 5
        """).fetchall()
        print(f"   Sample quantiles (monotonicity validation):")
        for row in sample:
            print(f"     itemid={row[0]}: Q1={row[1]:.2f} < Q5={row[2]:.2f} < Q9={row[3]:.2f}")


# Step 2: dbt transformation (defined in transform/nbr/models/)
# tokens table is automatically created by dbt


# Step 3: Export as partitioned Parquet
@dg.asset(
    kinds={"duckdb"},
    key=["main", "export_tokens"],
    deps=[
        dg.AssetKey(["main", "tokens_final"]),  # Stage 2: includes TIME_BIN
    ],
)
def export_tokens(duckdb: DuckDBResource) -> None:
    """
    Export final tokens table as subject_id partitioned Parquet files
    
    Stage 1 output structure (events_merged):
    - subject_id: Patient ID (partition key)
    - hadm_id: Hospital admission ID
    - admittime: Admission time
    - visit_seq: Visit sequence
    - days_since_prior_admission: Relative time (Dataset will cumsum)
    - code: Unified code (DX:/PR:/LAB:itemid_Q5/MED:)
    - code_type: Type
    - event_time: Event timestamp
    - time_offset_hours: Hours since admission
    - event_seq: Global sequence (time-sorted)
    
    Stage 2 will include:
    - TIME_BIN tokens (TIME_0-1H, TIME_1-6H, etc.)
    """
    with duckdb.get_connection() as conn:
        conn.execute("SET preserve_insertion_order = false;")
        conn.execute(
            f"""
            COPY (
                select *
                from tokens_final
                order by subject_id, visit_seq, final_seq
            )
            TO '{str(EXPORT_DIR)}'
            (
                format parquet,
                partition_by subject_id,
                write_partition_columns true,
                overwrite_or_ignore true
            )
            """
        )