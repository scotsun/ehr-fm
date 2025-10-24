
  
  create view "mimic4"."main"."stg_diagnoses__dbt_tmp" as (
    

/*
Staging: Diagnoses (诊断)

功能:
- 添加前缀 'DX:' 区分诊断编码
- 保留原始 seq_num（诊断优先级顺序）
- 诊断没有精确时间戳（event_time=NULL）

输出:
- 每行 = 一个诊断编码
- 关联到 hadm_id
*/

select 
    subject_id,
    hadm_id,
    'DX:' || icd_code as code,
    'diagnosis' as code_type,
    null::timestamp as event_time,  -- 诊断无精确时间
    seq_num
from "mimic4"."main"."raw_diagnoses_icd"
where icd_code is not null
  );
