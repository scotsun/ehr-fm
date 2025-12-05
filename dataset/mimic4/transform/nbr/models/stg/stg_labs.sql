{{
    config(
        materialized='view'
    )
}}

/*
Staging: Lab Events (ETHOS-aligned)

Functions:
1. Top-200 filtering: Only keep the most common 200 lab types
2. "Reclaim" strategy: Retrieve labs from 24 hours before admission (custom enhancement)
3. Value binning: Q1-Q10 quantile discretization
4. Filter out NULL values (ETHOS: df.valuenum.notna())

ETHOS Strategy (preprocessors.py:394-438):
- Each lab result becomes TWO tokens: LAB:{itemid} and _Q{1-10}
- Both tokens share the same timestamp (will be flattened in events_merged)
- NULL values are DISCARDED (not tokenized)

Output:
- Two rows per lab result (same charttime)
- Row 1: code = LAB:{itemid}, code_type = 'lab'
- Row 2: code = _Q{1-10}, code_type = 'lab_value'
*/

-- Step 1: Extend lab data (original + reclaimed)
with labs_original as (
    select
        subject_id,
        hadm_id,
        itemid,
        charttime,
        valuenum
    from {{ source('main', 'raw_labevents') }}
    where hadm_id is not null
        and valuenum is not null  -- ETHOS: filter out NULL values
        and itemid in (select itemid from {{ source('main', 'top_200_labs') }})
),

labs_reclaimed_raw as (
    -- "Reclaim" candidates: labs from 24 hours before admission
    select
        l.labevent_id,
        l.subject_id,
        a.hadm_id,
        l.itemid,
        l.charttime,
        l.valuenum,
        a.admittime,
        abs(extract(epoch from (l.charttime - a.admittime))) as time_diff_seconds
    from {{ source('main', 'raw_labevents') }} l
    join {{ ref('stg_admissions') }} a
        on l.subject_id = a.subject_id
        and l.hadm_id is null
        and l.charttime >= (a.admittime - interval '24 hours')
        and l.charttime <= a.admittime
    where l.valuenum is not null  -- ETHOS: filter out NULL values
        and l.itemid in (select itemid from {{ source('main', 'top_200_labs') }})
),

labs_reclaimed as (
    -- Each lab only assigned to nearest admission
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
    select * from labs_original
    union all
    select * from labs_reclaimed
),

-- Step 2: Value binning (Q1-Q10)
-- Note: NULL values already filtered out in Step 1
labs_binned as (
    select
        l.subject_id,
        l.hadm_id,
        l.itemid,
        l.charttime,
        case
            when l.valuenum <= q.q1 then '_Q1'
            when l.valuenum <= q.q2 then '_Q2'
            when l.valuenum <= q.q3 then '_Q3'
            when l.valuenum <= q.q4 then '_Q4'
            when l.valuenum <= q.q5 then '_Q5'
            when l.valuenum <= q.q6 then '_Q6'
            when l.valuenum <= q.q7 then '_Q7'
            when l.valuenum <= q.q8 then '_Q8'
            when l.valuenum <= q.q9 then '_Q9'
            else '_Q10'
        end as quantile_token
    from labs_extended l
    left join {{ source('main', 'lab_quantiles') }} q
        on l.itemid = q.itemid
),

-- Step 3: Split into two tokens per lab
-- Token 1: LAB:{itemid}
lab_tokens as (
    select
        subject_id,
        hadm_id,
        'LAB:' || itemid as code,
        'lab' as code_type,
        charttime as event_time,
        1 as token_order  -- Lab name comes first
    from labs_binned
),

-- Token 2: _Q{1-10}
quantile_tokens as (
    select
        subject_id,
        hadm_id,
        quantile_token as code,
        'lab_value' as code_type,
        charttime as event_time,
        2 as token_order  -- Quantile comes second
    from labs_binned
),

-- Combine both tokens
all_lab_tokens as (
    select * from lab_tokens
    union all
    select * from quantile_tokens
)

-- Final output with seq_num
select
    subject_id,
    hadm_id,
    code,
    code_type,
    event_time,
    row_number() over (
        partition by subject_id, hadm_id
        order by event_time, token_order
    ) as seq_num
from all_lab_tokens

