"""
MIMIC-IV Data Asset Definitions

Data Pipeline:
1. Import raw CSV to DuckDB (raw_* assets)
2. dbt transformation and merging (tokens models)
3. Export as partitioned Parquet (export_tokens asset)
"""

import dagster as dg
from dagster_duckdb import DuckDBResource
from mimic4.constants import RAW_DATA_DIR, MAPPING_DIR, EXPORT_DIR


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


@dg.asset(kinds={"duckdb"}, key=["main", "raw_emar"])
def raw_emar(duckdb: DuckDBResource) -> None:
    """
    Import emar.csv to raw_emar table

    Key fields:
    - subject_id: Patient ID
    - hadm_id: Hospital admission ID
    - medication: Drug name (used for ATC mapping)
    - charttime: Administration time
    - event_txt: Event type (filter for 'Administered')
    """
    with duckdb.get_connection() as conn:
        conn.execute(
            f"""
            create or replace table raw_emar as (
                select * from read_csv_auto('{RAW_DATA_DIR}/emar.csv', ignore_errors=true)
            )
            """
        )


# Mapping tables for code translation
@dg.asset(kinds={"duckdb"}, key=["main", "mapping_gsn_atc"])
def mapping_gsn_atc(duckdb: DuckDBResource) -> None:
    """
    Import GSN to ATC mapping table

    Source: gsn_atc_ndc_mapping.csv
    Key fields:
    - gsn: Generic Sequence Number
    - atc: ATC code
    """
    with duckdb.get_connection() as conn:
        conn.execute(
            f"""
            create or replace table mapping_gsn_atc as (
                select
                    cast(gsn as varchar) as gsn,
                    cast(atc as varchar) as atc
                from read_csv_auto('{MAPPING_DIR}/gsn_atc_ndc_mapping.csv')
            )
            """
        )


@dg.asset(kinds={"duckdb"}, key=["main", "mapping_icd_cm_9_to_10"])
def mapping_icd_cm_9_to_10(duckdb: DuckDBResource) -> None:
    """
    Import ICD-CM (diagnosis) 9 to 10 mapping table

    Source: icd_cm_9_to_10_mapping.csv
    Key fields:
    - icd_9: ICD-9 code
    - icd_10: ICD-10 code

    ETHOS strategy: When one ICD-9 maps to multiple ICD-10, take the shortest one
    """
    with duckdb.get_connection() as conn:
        conn.execute(
            f"""
            create or replace table mapping_icd_cm_9_to_10 as (
                select
                    cast(icd_9 as varchar) as icd_9,
                    cast(icd_10 as varchar) as icd_10
                from read_csv_auto('{MAPPING_DIR}/icd_cm_9_to_10_mapping.csv')
            )
            """
        )


@dg.asset(kinds={"duckdb"}, key=["main", "mapping_icd_pcs_9_to_10"])
def mapping_icd_pcs_9_to_10(duckdb: DuckDBResource) -> None:
    """
    Import ICD-PCS (procedure) 9 to 10 mapping table

    Source: icd_pcs_9_to_10_mapping.csv
    Key fields:
    - icd_9: ICD-9 procedure code
    - icd_10: ICD-10-PCS code

    ETHOS strategy: When one ICD-9 maps to multiple ICD-10, take the shortest one
    """
    with duckdb.get_connection() as conn:
        conn.execute(
            f"""
            create or replace table mapping_icd_pcs_9_to_10 as (
                select
                    cast(icd_9 as varchar) as icd_9,
                    cast(icd_10 as varchar) as icd_10
                from read_csv_auto('{MAPPING_DIR}/icd_pcs_9_to_10_mapping.csv')
            )
            """
        )


# Stage 1: Pre-computed lookup tables

@dg.asset(
    kinds={"duckdb"},
    key=["main", "drug_to_atc"],
    deps=[
        dg.AssetKey(["main", "raw_prescriptions"]),
        dg.AssetKey(["main", "mapping_gsn_atc"])
    ]
)
def drug_to_atc(duckdb: DuckDBResource) -> None:
    """
    Build drug name to ATC code lookup table

    ETHOS logic (code_translation.py:13-35):
    1. From prescriptions, get (drug, gsn) pairs
    2. GSN field may contain multiple space-separated codes -> split and explode
    3. Map GSN to ATC via gsn_atc_ndc_mapping
    4. drop_duplicates() then set_index('drug') -> last ATC wins for each drug
    5. dropna() -> discard drugs without valid ATC mapping

    Output:
    - drug: Drug name (from prescriptions)
    - atc_code: ATC code
    """
    with duckdb.get_connection() as conn:
        print("Building drug_to_atc lookup table...")
        conn.execute("""
            create or replace table drug_to_atc as
            with drug_gsn as (
                -- Step 1-2: Get drug-gsn pairs, split multiple GSNs
                select distinct
                    drug,
                    trim(unnest(string_split(cast(gsn as varchar), ' '))) as gsn
                from raw_prescriptions
                where drug is not null
                    and gsn is not null
            ),
            drug_atc_all as (
                -- Step 3: Map GSN to ATC
                select
                    dg.drug,
                    m.atc as atc_code
                from drug_gsn dg
                join mapping_gsn_atc m on dg.gsn = m.gsn
                where m.atc is not null
            ),
            drug_atc_dedup as (
                -- Step 4: Deduplicate - keep last occurrence per drug
                -- (simulates pandas set_index behavior)
                select
                    drug,
                    atc_code,
                    row_number() over (partition by drug order by atc_code desc) as rn
                from (select distinct drug, atc_code from drug_atc_all)
            )
            -- Step 5: Final lookup table
            select drug, atc_code
            from drug_atc_dedup
            where rn = 1
        """)

        # Validation
        result = conn.execute("select count(*) from drug_to_atc").fetchone()[0]
        print(f"✅ drug_to_atc lookup table created")
        print(f"   Unique drugs with ATC mapping: {result}")


@dg.asset(
    kinds={"duckdb"},
    key=["main", "icd_cm_9_to_10"],
    deps=[dg.AssetKey(["main", "mapping_icd_cm_9_to_10"])]
)
def icd_cm_9_to_10(duckdb: DuckDBResource) -> None:
    """
    Build ICD-CM (diagnosis) 9 to 10 lookup table

    ETHOS logic (translation_base.py:144-150):
    1. Load mapping file
    2. drop_duplicates(subset='icd_9')
    3. groupby('icd_9').icd_10.apply(lambda v: min(v, key=len))
       -> When one ICD-9 maps to multiple ICD-10, take the SHORTEST one

    Output:
    - icd_9: ICD-9 diagnosis code
    - icd_10: ICD-10 diagnosis code (shortest if multiple)
    """
    with duckdb.get_connection() as conn:
        print("Building icd_cm_9_to_10 lookup table...")
        conn.execute("""
            create or replace table icd_cm_9_to_10 as
            with ranked as (
                select
                    icd_9,
                    icd_10,
                    row_number() over (
                        partition by icd_9
                        order by length(icd_10), icd_10
                    ) as rn
                from mapping_icd_cm_9_to_10
                where icd_9 is not null and icd_10 is not null
            )
            select icd_9, icd_10
            from ranked
            where rn = 1
        """)

        # Validation
        result = conn.execute("select count(*) from icd_cm_9_to_10").fetchone()[0]
        print(f"✅ icd_cm_9_to_10 lookup table created")
        print(f"   Unique ICD-9 codes: {result}")


@dg.asset(
    kinds={"duckdb"},
    key=["main", "icd_pcs_9_to_10"],
    deps=[dg.AssetKey(["main", "mapping_icd_pcs_9_to_10"])]
)
def icd_pcs_9_to_10(duckdb: DuckDBResource) -> None:
    """
    Build ICD-PCS (procedure) 9 to 10 lookup table

    ETHOS logic (translation_base.py:144-150):
    Same as ICD-CM - when one ICD-9 maps to multiple ICD-10, take the SHORTEST one

    Output:
    - icd_9: ICD-9 procedure code
    - icd_10: ICD-10-PCS code (shortest if multiple)
    """
    with duckdb.get_connection() as conn:
        print("Building icd_pcs_9_to_10 lookup table...")
        conn.execute("""
            create or replace table icd_pcs_9_to_10 as
            with ranked as (
                select
                    icd_9,
                    icd_10,
                    row_number() over (
                        partition by icd_9
                        order by length(icd_10), icd_10
                    ) as rn
                from mapping_icd_pcs_9_to_10
                where icd_9 is not null and icd_10 is not null
            )
            select icd_9, icd_10
            from ranked
            where rn = 1
        """)

        # Validation
        result = conn.execute("select count(*) from icd_pcs_9_to_10").fetchone()[0]
        print(f"✅ icd_pcs_9_to_10 lookup table created")
        print(f"   Unique ICD-9 codes: {result}")


# Stage 2: Pre-computed tables (Lab processing)
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