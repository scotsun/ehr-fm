{{
    config(
        materialized='view'
    )
}}

/*
Staging: Admissions
- Calculates visit_seq and days_since_prior_admission
*/

select 
    subject_id,
    hadm_id,
    admittime,
    dischtime,
    -- Calculate visit sequence (group by patient, sort by admission time)
    row_number() over (
        partition by subject_id 
        order by admittime
    ) as visit_seq,
    
    -- Calculate days since last admission (relative time interval)
    cast(admittime as date) - lag(cast(admittime as date)) over (
        partition by subject_id 
        order by admittime
    ) as days_since_prior_admission

from {{ source('main', 'raw_admissions') }}
where hadm_id is not null  -- Filter invalid records

