{{
    config(
        materialized='table'
    )
}}

/*
Final Tokens with TIME_BIN
*/

with events as (
    select * from {{ ref('events_merged') }}
),

-- Calculate time bin and mark changes
events_with_bin as (
    select 
        *,
        case 
            when time_offset_hours < 0 then 'TIME_PRE'
            when time_offset_hours < 1 then 'TIME_0-1H'
            when time_offset_hours < 6 then 'TIME_1-6H'
            when time_offset_hours < 24 then 'TIME_6-24H'
            when time_offset_hours < 72 then 'TIME_1-3D'
            when time_offset_hours < 168 then 'TIME_3-7D'
            else 'TIME_7D+'
        end as time_bin,
        lag(case 
            when time_offset_hours < 0 then 'TIME_PRE'
            when time_offset_hours < 1 then 'TIME_0-1H'
            when time_offset_hours < 6 then 'TIME_1-6H'
            when time_offset_hours < 24 then 'TIME_6-24H'
            when time_offset_hours < 72 then 'TIME_1-3D'
            when time_offset_hours < 168 then 'TIME_3-7D'
            else 'TIME_7D+'
        end) over (partition by subject_id, hadm_id order by event_seq) as prev_bin
    from events
),

-- Generate TIME_BIN tokens
time_tokens as (
    select 
        subject_id, hadm_id, admittime, visit_seq, days_since_prior_admission,
        time_bin as code,
        'time_bin' as code_type,
        event_time,
        time_offset_hours,
        event_seq - 0.5 as insert_position
    from events_with_bin
    where time_bin != prev_bin or prev_bin is null
),

-- Merge
all_tokens as (
    select 
        subject_id, hadm_id, admittime, visit_seq, days_since_prior_admission,
        code, code_type, event_time, time_offset_hours,
        cast(event_seq as double) as sort_key
    from events
    
    union all
    
    select 
        subject_id, hadm_id, admittime, visit_seq, days_since_prior_admission,
        code, code_type, event_time, time_offset_hours,
        insert_position as sort_key
    from time_tokens
)

-- Final sorting
select 
    subject_id, hadm_id, admittime, visit_seq, days_since_prior_admission,
    code, code_type, event_time, time_offset_hours,
    row_number() over (partition by subject_id, hadm_id order by sort_key) as final_seq
from all_tokens
order by subject_id, visit_seq, final_seq

