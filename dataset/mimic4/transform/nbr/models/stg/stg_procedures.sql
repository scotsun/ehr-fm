{{
    config(
        materialized='view'
    )
}}

/*
Staging: Procedures (ETHOS-aligned)

ETHOS Logic (preprocessors.py:313-339):
1. Check icd_version: if 9, convert to ICD-10-PCS
2. Time: use chartdate (procedure date)
3. Preserve seq_num for procedure priority

ICD-9 to ICD-10 conversion (translation_base.py:144-150):
- When one ICD-9 maps to multiple ICD-10, take the SHORTEST one

Output:
- Each row = one procedure code (ICD-10-PCS)
- code format: PR:{icd10_code}
- event_time = chartdate
*/

with procedures_raw as (
    select
        p.subject_id,
        p.hadm_id,
        p.icd_code,
        p.icd_version,
        p.chartdate,
        p.seq_num
    from {{ source('main', 'raw_procedures_icd') }} p
    where p.icd_code is not null
        and p.hadm_id is not null
),

-- Convert ICD-9 to ICD-10 where needed
procedures_converted as (
    select
        pr.subject_id,
        pr.hadm_id,
        case
            when pr.icd_version = 9 then m.icd_10
            else pr.icd_code
        end as icd_code,
        pr.chartdate,
        pr.seq_num
    from procedures_raw pr
    left join {{ source('main', 'icd_pcs_9_to_10') }} m
        on pr.icd_version = 9 and pr.icd_code = m.icd_9
)

select
    subject_id,
    hadm_id,
    'PR:' || icd_code as code,
    'procedure' as code_type,
    cast(chartdate as timestamp) as event_time,
    seq_num
from procedures_converted
where icd_code is not null  -- Drop procedures where ICD-9 couldn't be converted

