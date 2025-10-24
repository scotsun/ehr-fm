

/*
Staging: Admissions (就诊主表)

功能:
- 计算就诊序号（visit_seq）
- 计算距上次就诊天数（days_since_prior_admission）

输出:
- 每行 = 一次就诊
- 包含患者ID、就诊ID、时间特征
*/

select 
    subject_id,
    hadm_id,
    admittime,
    dischtime,
    -- 计算就诊序号（按患者分组，按入院时间排序）
    row_number() over (
        partition by subject_id 
        order by admittime
    ) as visit_seq,
    
    -- 计算距上次就诊的天数（相对时间间隔）
    cast(admittime as date) - lag(cast(admittime as date)) over (
        partition by subject_id 
        order by admittime
    ) as days_since_prior_admission

from "mimic4"."main"."raw_admissions"
where hadm_id is not null  -- 过滤无效记录