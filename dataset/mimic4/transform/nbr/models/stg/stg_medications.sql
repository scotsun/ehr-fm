{{
    config(
        materialized='view'
    )
}}

/*
Staging: Medications (ETHOS-aligned)
- Source: emar (administered meds), NOT prescriptions
- Maps medication name to ATC code
- Output: MED:{atc_code} with charttime
*/

select
    e.subject_id,
    e.hadm_id,
    'MED:' || d.atc_code as code,
    'medication' as code_type,
    e.charttime as event_time,
    row_number() over (
        partition by e.subject_id, e.hadm_id
        order by e.charttime
    ) as seq_num
from {{ source('main', 'raw_emar') }} e
inner join {{ source('main', 'drug_to_atc') }} d
    on e.medication = d.drug
where e.event_txt = 'Administered'
    and e.medication is not null
    and e.hadm_id is not null

