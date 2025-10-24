{{
    config(
        materialized='view'
    )
}}

/*
Staging: Lab Events - Most complex

Functions:
1. Top-200 filtering: Only keep the most common 200 lab types
2. "Reclaim" strategy: Retrieve labs from 24 hours before admission
3. Value binning: Q1-Q10 quantile discretization (ETHOS method)

ETHOS Strategy:
- Continuous values → Discrete categories (Q1-Q10)
- Preserve NULL values as separate category
- Vocabulary: 200 itemid × 11 states = 2200 lab tokens

Output:
- Each row = one lab result (binned)
- Code format: LAB:itemid_Q5
- Includes precise charttime
*/

-- Step 1: Extend lab data (original + reclaimed)
with labs_original as (
    -- 1.1 Original inpatient labs with hadm_id
    select 
        subject_id,
        hadm_id,
        itemid,
        charttime,
        valuenum
    from {{ source('main', 'raw_labevents') }}
    where hadm_id is not null
        and itemid in (select itemid from {{ source('main', 'top_200_labs') }})
),

labs_reclaimed_raw as (
    -- 1.2 "Reclaim" candidates: labs from 24 hours before admission
    select 
        l.labevent_id,
        l.subject_id,
        a.hadm_id,  -- Assign hadm_id
        l.itemid,
        l.charttime,
        l.valuenum,
        a.admittime,
        abs(extract(epoch from (l.charttime - a.admittime))) as time_diff_seconds
    from {{ source('main', 'raw_labevents') }} l
    join {{ ref('stg_admissions') }} a 
        on l.subject_id = a.subject_id
        and l.hadm_id is null  -- Originally had no hadm_id
        and l.charttime >= (a.admittime - interval '24 hours')
        and l.charttime <= a.admittime
    where l.itemid in (select itemid from {{ source('main', 'top_200_labs') }})
),

labs_reclaimed as (
    -- 1.3 Each lab only assigned to nearest admission
    select 
        subject_id,
        hadm_id,
        itemid,
        charttime,
        valuenum
    from (
        select 
            *,
            row_number() over (partition by labevent_id order by time_diff_seconds) as rn
        from labs_reclaimed_raw
    ) ranked
    where rn = 1
),

labs_extended as (
    -- 1.4 Merge original and reclaimed
    select * from labs_original
    union all
    select * from labs_reclaimed
),

-- Step 2: Value binning (Q1-Q10)
labs_binned as (
    select 
        l.subject_id,
        l.hadm_id,
        l.itemid,
        l.charttime,
        l.valuenum,
        -- Binning logic (11 categories)
        case 
            when l.valuenum is null then 'LAB:' || l.itemid || '_NULL'
            when l.valuenum <= q.q1 then 'LAB:' || l.itemid || '_Q1'
            when l.valuenum <= q.q2 then 'LAB:' || l.itemid || '_Q2'
            when l.valuenum <= q.q3 then 'LAB:' || l.itemid || '_Q3'
            when l.valuenum <= q.q4 then 'LAB:' || l.itemid || '_Q4'
            when l.valuenum <= q.q5 then 'LAB:' || l.itemid || '_Q5'
            when l.valuenum <= q.q6 then 'LAB:' || l.itemid || '_Q6'
            when l.valuenum <= q.q7 then 'LAB:' || l.itemid || '_Q7'
            when l.valuenum <= q.q8 then 'LAB:' || l.itemid || '_Q8'
            when l.valuenum <= q.q9 then 'LAB:' || l.itemid || '_Q9'
            else 'LAB:' || l.itemid || '_Q10'
        end as code
    from labs_extended l
    left join {{ source('main', 'lab_quantiles') }} q 
        on l.itemid = q.itemid
)

-- Step 3: Final output
select 
    subject_id,
    hadm_id,
    code,
    'lab' as code_type,
    charttime as event_time,  -- Lab time (second precision)
    row_number() over (
        partition by subject_id, hadm_id 
        order by charttime
    ) as seq_num
from labs_binned

