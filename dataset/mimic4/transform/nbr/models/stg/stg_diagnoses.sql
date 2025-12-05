{{
    config(
        materialized='view'
    )
}}

/*
Staging: Diagnoses (ETHOS-aligned)

ETHOS Logic (preprocessors.py:270-310):
1. Check icd_version: if 9, convert to ICD-10
2. Preserve seq_num for diagnosis priority

ICD-9 to ICD-10 conversion (translation_base.py:144-150):
- When one ICD-9 maps to multiple ICD-10, take the SHORTEST one

Custom Design:
- event_time = NULL (diagnosis has no actual timestamp in MIMIC-IV)
- events_merged.sql assigns time_offset_hours = -0.001h
- This ensures diagnoses appear BEFORE other admission events

Output:
- Each row = one diagnosis code (ICD-10)
- code format: DX:{icd10_code}
- event_time = NULL
*/

with diagnoses_raw as (
    select
        d.subject_id,
        d.hadm_id,
        d.icd_code,
        d.icd_version,
        d.seq_num
    from {{ source('main', 'raw_diagnoses_icd') }} d
    where d.icd_code is not null
        and d.hadm_id is not null
),

-- Convert ICD-9 to ICD-10 where needed
diagnoses_converted as (
    select
        dr.subject_id,
        dr.hadm_id,
        case
            when dr.icd_version = 9 then m.icd_10
            else dr.icd_code
        end as icd_code,
        dr.seq_num
    from diagnoses_raw dr
    left join {{ source('main', 'icd_cm_9_to_10') }} m
        on dr.icd_version = 9 and dr.icd_code = m.icd_9
),

-- Join with admissions to validate hadm_id exists
diagnoses_validated as (
    select
        dc.subject_id,
        dc.hadm_id,
        dc.icd_code,
        dc.seq_num
    from diagnoses_converted dc
    join {{ ref('stg_admissions') }} a
        on dc.hadm_id = a.hadm_id
    where dc.icd_code is not null  -- Drop diagnoses where ICD-9 couldn't be converted
)

select
    subject_id,
    hadm_id,
    'DX:' || icd_code as code,
    'diagnosis' as code_type,
    cast(null as timestamp) as event_time,  -- No actual timestamp, will be -0.001h
    seq_num
from diagnoses_validated

