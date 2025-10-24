/*
MIMIC-IV Tokens 模型

功能：将多个医疗编码表合并为统一的 tokens 表
结构：subject_id → hadm_id (visit) → codes

类比 Instacart:
- subject_id = user_id
- hadm_id = order_id
- code = product_id (但需要加前缀区分类型)
*/

with admissions as (
    -- 就诊主表，计算就诊序号和时间间隔
    select 
        subject_id,
        hadm_id,
        admittime,
        dischtime,
        row_number() over (
            partition by subject_id 
            order by admittime
        ) as visit_seq,
        -- 计算距上次就诊的天数（类似 days_since_prior_order）
        CAST(admittime AS DATE) - lag(CAST(admittime AS DATE)) over (
            partition by subject_id 
            order by admittime
        ) as days_since_prior_admission
    from "mimic4"."main"."raw_admissions"
    where hadm_id is not null  -- 过滤无效记录
),

diagnoses as (
    -- 诊断编码，加前缀 "DX:"
    select 
        subject_id,
        hadm_id,
        'DX:' || icd_code as code,
        'diagnosis' as code_type,
        seq_num
    from "mimic4"."main"."raw_diagnoses_icd"
    where icd_code is not null
),

medications as (
    -- 药物编码，加前缀 "MED:"
    select 
        subject_id,
        hadm_id,
        'MED:' || drug as code,
        'medication' as code_type,
        row_number() over (
            partition by subject_id, hadm_id 
            order by starttime
        ) as seq_num
    from "mimic4"."main"."raw_prescriptions"
    where drug is not null
),

all_codes as (
    -- 合并所有编码
    select * from diagnoses
    union all
    select * from medications
)

-- 最终输出：每行代表某患者某次就诊的某个编码
select 
    a.subject_id,
    a.hadm_id,
    a.admittime,
    a.visit_seq,
    a.days_since_prior_admission,  -- 时间间隔
    c.code,
    c.code_type,
    c.seq_num
from admissions a
join all_codes c 
    on a.hadm_id = c.hadm_id
order by 
    a.subject_id, 
    a.admittime, 
    c.seq_num