
  
  create view "mimic4"."main"."stg_medications__dbt_tmp" as (
    

/*
Staging: Medications (用药)

功能:
- 添加前缀 'MED:' 区分药物
- **不去重**：保留所有用药记录（包括重复用药）
- 按 starttime 排序编号

ETHOS 策略:
- 输入(X): 保留所有用药事件，学习用药频率
- 目标(Y): 训练时按 hadm_id 去重（在 Dataset 中处理）

输出:
- 每行 = 一个用药记录
- 包含精确的 starttime（秒级时间）
*/

select 
    subject_id,
    hadm_id,
    'MED:' || drug as code,
    'medication' as code_type,
    starttime as event_time,  -- 用药开始时间（秒级精度）
    row_number() over (
        partition by subject_id, hadm_id 
        order by starttime
    ) as seq_num
from "mimic4"."main"."raw_prescriptions"
where drug is not null 
    and hadm_id is not null
  );
