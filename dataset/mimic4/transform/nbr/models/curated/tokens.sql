{{
    config(
        materialized='table'
    )
}}

/*
Final Tokens
- Outputs events_merged with final_seq for sequential position
- time_offset_hours preserved for RoPE encoding
*/

select
    subject_id,
    hadm_id,
    admittime,
    visit_seq,
    days_since_prior_admission,
    code,
    code_type,
    event_time,
    time_offset_hours,
    event_seq as final_seq
from {{ ref('events_merged') }}
order by subject_id, visit_seq, final_seq

