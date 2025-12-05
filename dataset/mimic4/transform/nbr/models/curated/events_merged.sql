{{
    config(
        materialized='table'
    )
}}

/*
Curated: Events Merged
- Merges DX + PR + LAB + MED events
- Time offset strategy:
  - Diagnoses: -0.001h (first)
  - Procedures before admission: 0.001h (after diagnoses)
  - Others: actual time offset from admittime
*/

-- Step 1: Merge all events
with all_events as (
    select * from {{ ref('stg_diagnoses') }}
    union all
    select * from {{ ref('stg_procedures') }}
    union all
    select * from {{ ref('stg_labs') }}
    union all
    select * from {{ ref('stg_medications') }}
),

-- Step 2: Join admission info and calculate time offset
events_with_time as (
    select
        a.subject_id,
        a.hadm_id,
        a.admittime,
        a.visit_seq,
        a.days_since_prior_admission,
        e.code,
        e.code_type,
        e.event_time,
        e.seq_num,

        -- Calculate time offset (hours)
        case
            -- Diagnoses: no time, place first (-0.001h)
            when e.event_time is null then -0.001
            -- Procedures before admission: place after diagnoses (0.001h)
            when e.code_type = 'procedure' and e.event_time < a.admittime then 0.001
            -- Others: actual time offset
            else extract(epoch from (e.event_time - a.admittime)) / 3600.0
        end as time_offset_hours

    from {{ ref('stg_admissions') }} a
    join all_events e on a.hadm_id = e.hadm_id
),

-- Step 3: Global time sorting and renumbering
events_sorted as (
    select 
        *,
        -- Global sequence (sorted by time)
        row_number() over (
            partition by subject_id, hadm_id 
            order by 
                time_offset_hours,  -- First by time
                seq_num,            -- Then by seq_num (for diagnoses/procedures priority)
                code                -- Finally by code (deterministic for same time/seq_num)
        ) as event_seq
    from events_with_time
)

-- Final output
select 
    subject_id,
    hadm_id,
    admittime,
    visit_seq,
    days_since_prior_admission,  -- relative time
    code,
    code_type,
    event_time,
    time_offset_hours,
    event_seq
from events_sorted
order by 
    subject_id, 
    visit_seq, 
    event_seq

