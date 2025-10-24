

/*
Staging: Procedures (手术/操作)

功能:
- 添加前缀 'PR:' 区分手术编码
- 将 chartdate 转换为 timestamp（用于时间排序）
- 保留原始 seq_num

输出:
- 每行 = 一个手术/操作编码
- 包含操作日期（日期级别的时间）
*/

select 
    subject_id,
    hadm_id,
    'PR:' || icd_code as code,
    'procedure' as code_type,
    cast(chartdate as timestamp) as event_time,  -- 日期转时间戳（00:00:00）
    seq_num
from "mimic4"."main"."raw_procedures_icd"
where icd_code is not null
    and hadm_id is not null