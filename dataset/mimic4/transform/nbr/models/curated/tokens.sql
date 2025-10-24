/*
MIMIC-IV Tokens Model

Function: Merge multiple medical coding tables into unified tokens table
Structure: subject_id → hadm_id (visit) → codes

Analogy to Instacart:
- subject_id = user_id
- hadm_id = order_id
- code = product_id (but needs prefix to distinguish types)
*/

with admissions as (
    -- Admission main table, calculate visit sequence and time intervals
    select 
        subject_id,
        hadm_id,
        admittime,
        dischtime,
        row_number() over (
            partition by subject_id 
            order by admittime
        ) as visit_seq,
        -- Calculate days since last admission (similar to days_since_prior_order)
        CAST(admittime AS DATE) - lag(CAST(admittime AS DATE)) over (
            partition by subject_id 
            order by admittime
        ) as days_since_prior_admission
    from {{ source('main', 'raw_admissions') }}
    where hadm_id is not null  -- Filter invalid records
),

diagnoses as (
    -- Diagnosis codes, add prefix "DX:"
    select 
        subject_id,
        hadm_id,
        'DX:' || icd_code as code,
        'diagnosis' as code_type,
        seq_num
    from {{ source('main', 'raw_diagnoses_icd') }}
    where icd_code is not null
),

medications as (
    -- Medication codes, add prefix "MED:"
    select 
        subject_id,
        hadm_id,
        'MED:' || drug as code,
        'medication' as code_type,
        row_number() over (
            partition by subject_id, hadm_id 
            order by starttime
        ) as seq_num
    from {{ source('main', 'raw_prescriptions') }}
    where drug is not null
),

all_codes as (
    -- Merge all codes
    select * from diagnoses
    union all
    select * from medications
)

-- Final output: each row represents a code for a patient's visit
select 
    a.subject_id,
    a.hadm_id,
    a.admittime,
    a.visit_seq,
    a.days_since_prior_admission,  -- Time interval
    c.code,
    c.code_type,
    c.seq_num
from admissions a
join all_codes c 
    on a.hadm_id = c.hadm_id
order by 
    a.subject_id, 
    a.admittime, 
    c.seq_num

