{{
    config(
        materialized='view'
    )
}}

/*
Staging: Diagnoses

Functions:
- Add prefix 'DX:' to distinguish diagnosis codes
- Preserve original seq_num (diagnosis priority order)
- Diagnoses have no precise timestamp (event_time=NULL)

Output:
- Each row = one diagnosis code
- Linked to hadm_id
*/

select 
    subject_id,
    hadm_id,
    'DX:' || icd_code as code,
    'diagnosis' as code_type,
    null::timestamp as event_time,  -- Diagnoses have no precise time
    seq_num
from {{ source('main', 'raw_diagnoses_icd') }}
where icd_code is not null

