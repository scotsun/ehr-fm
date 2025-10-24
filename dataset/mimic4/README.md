# ETHOS Pipeline Implementation Guide - 2 Stages

## Stage 1 Goal

Create pre-computed tables and validate the basic data flow (excluding TIME\_BIN)

-----

## Files Created

### Assets (src/mimic4/defs/assets.py)

  - `raw_procedures_icd` - Import procedures table
  - `raw_labevents` - Import lab events table (17GB, ignore\_errors)
  - `top_200_labs` - Calculate Top-200 lab items
  - `lab_quantiles` - Calculate quantiles (sampling 10%)

### dbt Models (transform/nbr/models/)

  - `stg/stg_admissions.sql` - Main admissions table (visit\_seq, days\_since\_prior)
  - `stg/stg_diagnoses.sql` - Diagnoses (DX: prefix, no time)
  - `stg/stg_procedures.sql` - Procedures (PR: prefix, chartdate)
  - `stg/stg_medications.sql` - Medications (MED: prefix, no deduplication, starttime)
  - `stg/stg_labs.sql` - Labs (LAB:itemid\_Q1-Q10, backfill + binning)
  - `curated/events_merged.sql` - Merge all events, global time sort

-----

## Run Steps

### Step 1: Start Dagster UI

```bash
cd /Users/gin/Desktop/HAT/ehr-fm/dataset/mimic4
uv sync
source .venv/bin/activate
dg dev --port 3001
```

Open in browser: http://localhost:3001

-----

### Step 2: Materialize Assets in Order

**Important: Must be executed in order\!**

#### 2.1 Import Raw Tables

1.  `main/raw_admissions`
2.  `main/raw_diagnoses_icd`
3.  `main/raw_prescriptions`
4.  `main/raw_procedures_icd`
5.  `main/raw_labevents`

-----

#### 2.2 Pre-computation Tables

6.  `main/top_200_labs`
7.  `main/lab_quantiles`

-----

#### 2.3 dbt Models

8.  `main/stg_admissions`
9.  `main/stg_diagnoses`
10. `main/stg_procedures`
11. `main/stg_medications`
12. `main/stg_labs`
13. `main/events_merged`

**Key Features:**

  - `event_seq` is strictly increasing (1, 2, 3...)
  - `time_offset_hours` is generally increasing (Diagnoses at -0.001 first, then by actual time)
  - `code` includes prefix and quantile (e.g., LAB:itemid\_Q5)
  - `days_since_prior` is retained (Relative time, for Dataset cumsum)

-----

## Next Steps (already done)

After Stage 1 is successful:

1.  **Validate data quality**
2.  **Analyze actual sequence length**
3.  **Proceed to Stage 2**: Add `TIME_BIN` tokens
4.  **Export to Parquet**