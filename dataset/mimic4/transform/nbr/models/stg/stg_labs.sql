{{
    config(
        materialized='view'
    )
}}

/*
Staging: Lab Events
- Top-200 labs, NULL values discarded
- Reclaims labs from 24h before admission
- Output: LAB:{itemid}_Q{1-10}
*/

with labs_original as (
    select
        l.subject_id,
        l.hadm_id,
        l.itemid,
        l.charttime,
        l.valuenum
    from {{ source('main', 'raw_labevents') }} l
    join {{ ref('stg_admissions') }} a
        on l.hadm_id = a.hadm_id
        and l.subject_id = a.subject_id
    where l.hadm_id is not null
        and l.valuenum is not null
        and l.itemid in (select itemid from {{ source('main', 'top_200_labs') }})
        and l.charttime >= (a.admittime - interval '24 hours')
),

labs_reclaimed_raw as (
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
    where l.valuenum is not null
        and l.itemid in (select itemid from {{ source('main', 'top_200_labs') }})
),

labs_reclaimed as (
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

labs_binned as (
    select
        l.subject_id,
        l.hadm_id,
        l.itemid,
        l.charttime,
        'LAB:' || l.itemid || case
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
        end as code
    from labs_extended l
    left join {{ source('main', 'lab_quantiles') }} q
        on l.itemid = q.itemid
)

select
    subject_id,
    hadm_id,
    code,
    'lab' as code_type,
    charttime as event_time,
    row_number() over (
        partition by subject_id, hadm_id
        order by charttime
    ) as seq_num
from labs_binned

