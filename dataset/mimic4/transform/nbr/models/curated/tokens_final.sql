{{
    config(
        materialized='table'
    )
}}

/*
Final Tokens (No TIME_BIN)

Simplified version: directly outputs events from events_merged.
Time information is preserved via time_offset_hours for RoPE encoding.

Output columns:
- subject_id, hadm_id: patient and encounter identifiers
- admittime, visit_seq, days_since_prior_admission: encounter context
- code, code_type: token code and category
- event_time, time_offset_hours: precise timing for RoPE
- final_seq: sequential position within encounter
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

