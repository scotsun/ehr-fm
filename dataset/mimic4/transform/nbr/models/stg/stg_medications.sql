{{
    config(
        materialized='view'
    )
}}

/*
Staging: Medications

Functions:
- Add prefix 'MED:' to distinguish medications
- **No deduplication**: Keep all medication records (including duplicates)
- Sort and number by starttime

ETHOS Strategy:
- Input (X): Keep all medication events, learn medication frequency
- Target (Y): Deduplicate by hadm_id during training (handled in Dataset)

Output:
- Each row = one medication record
- Includes precise starttime (second precision)
*/

select 
    subject_id,
    hadm_id,
    'MED:' || drug as code,
    'medication' as code_type,
    starttime as event_time,  -- Medication start time (second precision)
    row_number() over (
        partition by subject_id, hadm_id 
        order by starttime
    ) as seq_num
from {{ source('main', 'raw_prescriptions') }}
where drug is not null 
    and hadm_id is not null

