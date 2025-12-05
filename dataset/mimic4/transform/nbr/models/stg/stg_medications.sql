{{
    config(
        materialized='view'
    )
}}

/*
Staging: Medications (ETHOS-aligned)

Data Source: emar (actual administered medications)
- NOT prescriptions (which is only used for drug→ATC mapping)

ETHOS Logic (preprocessors.py:363-391):
1. Filter: event_txt = 'Administered' and medication is not null
2. Map: medication → ATC code via drug_to_atc lookup table
3. Time: charttime (actual administration time)
4. Discard: medications without valid ATC mapping (inner join)

Output:
- Each row = one administered medication
- code format: MED:{atc_code}
- Includes precise charttime
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

