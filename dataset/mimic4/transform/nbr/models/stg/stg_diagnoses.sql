{{
    config(
        materialized='view'
    )
}}

/*
Staging: Diagnoses (ETHOS-aligned)
- Converts ICD-9 to ICD-10 (shortest match when multiple)
- event_time = NULL (placed at -0.001h in events_merged)
- Output: DX:{icd10_code}
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
)

-- Final output (hadm_id validation done in events_merged.sql)
select
    subject_id,
    hadm_id,
    'DX:' || icd_code as code,
    'diagnosis' as code_type,
    cast(null as timestamp) as event_time,  -- No actual timestamp, will be -0.001h
    seq_num
from diagnoses_converted
where icd_code is not null  -- Drop diagnoses where ICD-9 couldn't be converted

