{{
    config(
        materialized='view'
    )
}}

/*
Staging: Procedures

Functions:
- Add prefix 'PR:' to distinguish procedure codes
- Convert chartdate to timestamp (for time sorting)
- Preserve original seq_num

Output:
- Each row = one procedure/operation code
- Includes operation date (date-level time)
*/

select 
    subject_id,
    hadm_id,
    'PR:' || icd_code as code,
    'procedure' as code_type,
    cast(chartdate as timestamp) as event_time,  -- Convert date to timestamp (00:00:00)
    seq_num
from {{ source('main', 'raw_procedures_icd') }}
where icd_code is not null
    and hadm_id is not null

