

/*
Staging: Lab Events (检验结果) - 最复杂的处理

功能:
1. Top-200 过滤：只保留最常见的 200 种检验
2. "捞回"策略：找回入院前 24 小时的检验
3. 数值分箱：Q1-Q10 分位数离散化（ETHOS 方法）

ETHOS 策略:
- 连续数值 → 离散类别（Q1-Q10）
- 保留 NULL 值作为单独类别
- 词表: 200 itemid × 11 状态 = 2200 个 lab tokens

输出:
- 每行 = 一个检验结果（已分箱）
- code 格式: LAB:itemid_Q5
- 包含精确 charttime
*/

-- ============================================================================
-- Step 1: 扩展检验数据（原有 + 捞回）
-- ============================================================================
with labs_original as (
    -- 1.1 原本有 hadm_id 的住院检验
    select 
        subject_id,
        hadm_id,
        itemid,
        charttime,
        valuenum
    from "mimic4"."main"."raw_labevents"
    where hadm_id is not null
        and itemid in (select itemid from "mimic4"."main"."top_200_labs")
),

labs_reclaimed_raw as (
    -- 1.2 "捞回"候选：入院前 24 小时的检验
    select 
        l.labevent_id,
        l.subject_id,
        a.hadm_id,  -- 分配 hadm_id
        l.itemid,
        l.charttime,
        l.valuenum,
        a.admittime,
        abs(extract(epoch from (l.charttime - a.admittime))) as time_diff_seconds
    from "mimic4"."main"."raw_labevents" l
    join "mimic4"."main"."stg_admissions" a 
        on l.subject_id = a.subject_id
        and l.hadm_id is null  -- 原本没有 hadm_id
        and l.charttime >= (a.admittime - interval '24 hours')
        and l.charttime <= a.admittime
    where l.itemid in (select itemid from "mimic4"."main"."top_200_labs")
),

labs_reclaimed as (
    -- 1.3 每个检验只分配给最近的就诊
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
    -- 1.4 合并原有和捞回的
    select * from labs_original
    union all
    select * from labs_reclaimed
),

-- ============================================================================
-- Step 2: 数值分箱（Q1-Q10）
-- ============================================================================
labs_binned as (
    select 
        l.subject_id,
        l.hadm_id,
        l.itemid,
        l.charttime,
        l.valuenum,
        -- 分箱逻辑（11个类别）
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
    left join "mimic4"."main"."lab_quantiles" q 
        on l.itemid = q.itemid
)

-- ============================================================================
-- Step 3: 最终输出
-- ============================================================================
select 
    subject_id,
    hadm_id,
    code,
    'lab' as code_type,
    charttime as event_time,  -- 检验时间（秒级精度）
    row_number() over (
        partition by subject_id, hadm_id 
        order by charttime
    ) as seq_num
from labs_binned