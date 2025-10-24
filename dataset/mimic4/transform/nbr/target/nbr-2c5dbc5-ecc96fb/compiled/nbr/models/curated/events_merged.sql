

/*
Curated: Events Merged (事件合并 - ETHOS Pipeline 核心)

功能:
1. 合并 4 类事件（DX + PR + LAB + MED）
2. 计算时间偏移（距入院小时数）
3. 全局时间排序（ETHOS 关键特性）

ETHOS 策略:
- 所有事件按实际发生时间排序
- 不按类型分组
- 保留时间信息用于后续插入 TIME_BIN

输出:
- 每行 = 一个事件（诊断/手术/检验/用药）
- 按 subject_id, visit_seq, 时间 排序
- 包含 time_offset_hours 用于 TIME_BIN 生成
*/

-- ============================================================================
-- Step 1: 合并所有事件
-- ============================================================================
with all_events as (
    select * from "mimic4"."main"."stg_diagnoses"
    union all
    select * from "mimic4"."main"."stg_procedures"
    union all
    select * from "mimic4"."main"."stg_labs"
    union all
    select * from "mimic4"."main"."stg_medications"
),

-- ============================================================================
-- Step 2: 关联就诊信息并计算时间偏移
-- ============================================================================
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
        
        -- 计算时间偏移（小时）
        -- 策略：诊断无时间，放在最前面（-0.001小时）
        case 
            when e.event_time is null then -0.001
            else extract(epoch from (e.event_time - a.admittime)) / 3600.0
        end as time_offset_hours
        
    from "mimic4"."main"."stg_admissions" a
    join all_events e on a.hadm_id = e.hadm_id
),

-- ============================================================================
-- Step 3: 全局时间排序并重新编号
-- ============================================================================
events_sorted as (
    select 
        *,
        -- 全局序号（按时间排序）
        row_number() over (
            partition by subject_id, hadm_id 
            order by 
                time_offset_hours,  -- 首先按时间
                code                -- 时间相同时按 code 字典序（确定性）
        ) as event_seq
    from events_with_time
)

-- ============================================================================
-- 最终输出
-- ============================================================================
select 
    subject_id,
    hadm_id,
    admittime,
    visit_seq,
    days_since_prior_admission,  -- ← 保留相对时间（用于 Dataset cumsum）
    code,
    code_type,
    event_time,
    time_offset_hours,
    event_seq
from events_sorted
order by 
    subject_id, 
    visit_seq, 
    event_seq  -- ← ETHOS 关键：全局时间排序