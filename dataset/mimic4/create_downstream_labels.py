"""
Create downstream task labels from MIMIC-IV raw data.

Tasks:
1. Inpatient Mortality (binary) - from hospital_expire_flag
2. 30-day Readmission (binary) - next admission within 30 days
3. Prolonged LoS >7d (binary) - length of stay > 7 days
4. ICD Chapter Prediction (multi-class) - primary diagnosis chapter
5. Abnormal Lab Q1/Q10 (binary) - extreme lab values

Output: Labels CSV file with columns:
- subject_id, hadm_id, mortality, readmission_30d, prolonged_los, icd_chapter, has_abnormal_lab
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# Paths
RAW_DIR = Path(__file__).parent / "data" / "raw_mimic4"
OUTPUT_DIR = Path(__file__).parent / "data"
PARQUET_DIR = OUTPUT_DIR / "mimic4_tokens.parquet"


def get_icd_chapter(icd_code: str, version: int) -> int:
    """
    Map ICD code to chapter number (1-22 for ICD-10, 1-21 for ICD-9).
    Returns -1 if unknown.
    """
    if pd.isna(icd_code):
        return -1

    code = str(icd_code).strip().upper()

    if version == 10:
        # ICD-10-CM chapters based on first letter
        # https://www.icd10data.com/ICD10CM/Codes
        first_char = code[0] if code else ''

        # Handle D codes specially (spans two chapters)
        if first_char == 'D':
            try:
                # Extract numeric part after D
                numeric_part = ''.join(c for c in code[1:] if c.isdigit())
                if numeric_part:
                    num = int(numeric_part[:2])  # Get first two digits
                    if num < 50:
                        return 2   # D00-D49: Neoplasms (Chapter 2)
                    else:
                        return 3   # D50-D89: Blood diseases (Chapter 3)
            except (ValueError, IndexError):
                pass
            return 2  # Default D to Neoplasms if parsing fails

        # Handle H codes specially (spans two chapters)
        if first_char == 'H':
            try:
                numeric_part = ''.join(c for c in code[1:] if c.isdigit())
                if numeric_part:
                    num = int(numeric_part[:2])
                    if num < 60:
                        return 7   # H00-H59: Eye diseases (Chapter 7)
                    else:
                        return 8   # H60-H95: Ear diseases (Chapter 8)
            except (ValueError, IndexError):
                pass
            return 7  # Default H to Eye if parsing fails

        chapter_map = {
            'A': 1, 'B': 1,   # Infectious diseases
            'C': 2,           # Neoplasms
            'E': 4,           # Endocrine/metabolic
            'F': 5,           # Mental disorders
            'G': 6,           # Nervous system
            'I': 9,           # Circulatory system
            'J': 10,          # Respiratory system
            'K': 11,          # Digestive system
            'L': 12,          # Skin
            'M': 13,          # Musculoskeletal
            'N': 14,          # Genitourinary
            'O': 15,          # Pregnancy
            'P': 16,          # Perinatal
            'Q': 17,          # Congenital
            'R': 18,          # Symptoms/signs
            'S': 19, 'T': 19, # Injury/poisoning
            'V': 20, 'W': 20, 'X': 20, 'Y': 20,  # External causes
            'Z': 21,          # Health status/services
            'U': 22,          # Special purposes (COVID-19, etc.)
        }
        return chapter_map.get(first_char, -1)

    elif version == 9:
        # ICD-9-CM chapters based on numeric range
        try:
            # Remove leading zeros and non-numeric prefix
            numeric = ''.join(c for c in code if c.isdigit() or c == '.')
            if not numeric:
                # E-codes (external causes) and V-codes (supplementary)
                if code.startswith('E'):
                    return 19  # External causes
                elif code.startswith('V'):
                    return 21  # Supplementary
                return -1

            num = float(numeric.split('.')[0])

            if 1 <= num <= 139:
                return 1   # Infectious
            elif 140 <= num <= 239:
                return 2   # Neoplasms
            elif 240 <= num <= 279:
                return 4   # Endocrine
            elif 280 <= num <= 289:
                return 3   # Blood
            elif 290 <= num <= 319:
                return 5   # Mental
            elif 320 <= num <= 389:
                return 6   # Nervous
            elif 390 <= num <= 459:
                return 9   # Circulatory
            elif 460 <= num <= 519:
                return 10  # Respiratory
            elif 520 <= num <= 579:
                return 11  # Digestive
            elif 580 <= num <= 629:
                return 14  # Genitourinary
            elif 630 <= num <= 679:
                return 15  # Pregnancy
            elif 680 <= num <= 709:
                return 12  # Skin
            elif 710 <= num <= 739:
                return 13  # Musculoskeletal
            elif 740 <= num <= 759:
                return 17  # Congenital
            elif 760 <= num <= 779:
                return 16  # Perinatal
            elif 780 <= num <= 799:
                return 18  # Symptoms
            elif 800 <= num <= 999:
                return 19  # Injury
            else:
                return -1
        except (ValueError, IndexError):
            return -1

    return -1


def create_labels():
    print("=" * 60)
    print("Creating downstream task labels from MIMIC-IV")
    print("=" * 60)

    # Load raw data
    print("\n[1/6] Loading raw data...")
    admissions = pd.read_csv(
        RAW_DIR / "admissions.csv",
        parse_dates=['admittime', 'dischtime', 'deathtime'],
        usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime',
                 'deathtime', 'hospital_expire_flag']
    )
    print(f"  Admissions: {len(admissions):,} rows")

    diagnoses = pd.read_csv(
        RAW_DIR / "diagnoses_icd.csv",
        usecols=['subject_id', 'hadm_id', 'seq_num', 'icd_code', 'icd_version']
    )
    print(f"  Diagnoses: {len(diagnoses):,} rows")

    # Get patients in parquet (already processed)
    parquet_patients = set()
    for d in PARQUET_DIR.glob("subject_id=*"):
        parquet_patients.add(int(d.name.split('=')[1]))
    print(f"  Parquet patients: {len(parquet_patients):,}")

    # Filter to parquet patients only
    admissions = admissions[admissions['subject_id'].isin(parquet_patients)]
    print(f"  Filtered admissions: {len(admissions):,}")

    # Task 1: Inpatient Mortality
    print("\n[2/6] Creating mortality labels...")
    admissions['mortality'] = admissions['hospital_expire_flag'].astype(int)
    mortality_rate = admissions['mortality'].mean()
    print(f"  Mortality rate: {mortality_rate:.2%} ({admissions['mortality'].sum():,} / {len(admissions):,})")

    # Task 2: 30-day Readmission
    print("\n[3/6] Creating 30-day readmission labels...")
    admissions = admissions.sort_values(['subject_id', 'admittime'])

    # Calculate days to next admission
    admissions['next_admittime'] = admissions.groupby('subject_id')['admittime'].shift(-1)
    admissions['days_to_next'] = (admissions['next_admittime'] - admissions['dischtime']).dt.total_seconds() / 86400

    # Readmission within 30 days (exclude deaths and last admissions)
    admissions['readmission_30d'] = (
        (admissions['days_to_next'] <= 30) &
        (admissions['days_to_next'] >= 0) &  # Must be after discharge
        (admissions['mortality'] == 0)  # Exclude deaths
    ).astype(int)

    # Set -1 for ineligible (death or last admission)
    admissions.loc[admissions['mortality'] == 1, 'readmission_30d'] = -1
    admissions.loc[admissions['next_admittime'].isna(), 'readmission_30d'] = -1

    eligible = admissions[admissions['readmission_30d'] >= 0]
    readmission_rate = eligible['readmission_30d'].mean()
    print(f"  Readmission rate: {readmission_rate:.2%} ({eligible['readmission_30d'].sum():,} / {len(eligible):,})")
    print(f"  Ineligible (death/last admission): {(admissions['readmission_30d'] == -1).sum():,}")

    # Task 3: Prolonged LoS (>7 days)
    print("\n[4/6] Creating prolonged LoS labels...")
    admissions['los_days'] = (admissions['dischtime'] - admissions['admittime']).dt.total_seconds() / 86400
    admissions['prolonged_los'] = (admissions['los_days'] > 7).astype(int)

    prolonged_rate = admissions['prolonged_los'].mean()
    print(f"  Prolonged LoS rate: {prolonged_rate:.2%} ({admissions['prolonged_los'].sum():,} / {len(admissions):,})")
    print(f"  Mean LoS: {admissions['los_days'].mean():.1f} days, Median: {admissions['los_days'].median():.1f} days")

    # Task 4: ICD Chapter Prediction
    print("\n[5/6] Creating ICD chapter labels...")
    primary_dx = diagnoses[diagnoses['seq_num'] == 1].copy()
    primary_dx['icd_chapter'] = primary_dx.apply(
        lambda x: get_icd_chapter(x['icd_code'], x['icd_version']), axis=1
    )

    # Merge with admissions
    admissions = admissions.merge(
        primary_dx[['hadm_id', 'icd_chapter']],
        on='hadm_id',
        how='left'
    )
    admissions['icd_chapter'] = admissions['icd_chapter'].fillna(-1).astype(int)

    valid_chapters = admissions[admissions['icd_chapter'] >= 0]['icd_chapter']
    chapter_counts = Counter(valid_chapters)
    print(f"  Valid ICD chapters: {len(valid_chapters):,} / {len(admissions):,}")
    print(f"  Number of chapters: {len(chapter_counts)}")
    print(f"  Top 5 chapters: {chapter_counts.most_common(5)}")

    # Task 5: Abnormal Lab (Q1/Q10)
    # This task uses existing tokenization - we'll identify patients with extreme lab values
    print("\n[6/6] Creating abnormal lab labels...")
    print("  (This task uses existing Q1/Q10 tokens from parquet data)")
    print("  Label extraction will be done during dataset loading")
    admissions['has_abnormal_lab'] = -1  # Placeholder, computed at dataset level

    # Save labels
    print("\n" + "=" * 60)
    print("Saving labels...")

    labels = admissions[[
        'subject_id', 'hadm_id', 'admittime',
        'mortality', 'readmission_30d', 'prolonged_los', 'icd_chapter'
    ]].copy()

    output_path = OUTPUT_DIR / "downstream_labels.csv"
    labels.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    print(f"  Total rows: {len(labels):,}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Label Summary:")
    print("=" * 60)
    print(f"\n  Task 1 - Mortality:")
    print(f"    Positive: {labels['mortality'].sum():,} ({labels['mortality'].mean():.2%})")
    print(f"    Negative: {(labels['mortality'] == 0).sum():,} ({(labels['mortality'] == 0).mean():.2%})")

    print(f"\n  Task 2 - 30-day Readmission:")
    valid = labels[labels['readmission_30d'] >= 0]
    print(f"    Positive: {(valid['readmission_30d'] == 1).sum():,} ({valid['readmission_30d'].mean():.2%})")
    print(f"    Negative: {(valid['readmission_30d'] == 0).sum():,}")
    print(f"    Excluded: {(labels['readmission_30d'] == -1).sum():,}")

    print(f"\n  Task 3 - Prolonged LoS (>7d):")
    print(f"    Positive: {labels['prolonged_los'].sum():,} ({labels['prolonged_los'].mean():.2%})")
    print(f"    Negative: {(labels['prolonged_los'] == 0).sum():,}")

    print(f"\n  Task 4 - ICD Chapter:")
    valid_icd = labels[labels['icd_chapter'] >= 0]
    print(f"    Valid: {len(valid_icd):,}")
    print(f"    Invalid/Missing: {(labels['icd_chapter'] == -1).sum():,}")
    print(f"    Unique chapters: {labels['icd_chapter'].nunique() - 1}")  # -1 for invalid

    # Patient-level split recommendation
    print("\n" + "=" * 60)
    print("Patient-Level Split Recommendation:")
    print("=" * 60)
    unique_patients = labels['subject_id'].nunique()
    train_size = int(unique_patients * 0.7)
    val_size = int(unique_patients * 0.15)
    test_size = unique_patients - train_size - val_size
    print(f"  Total patients: {unique_patients:,}")
    print(f"  Train (70%): {train_size:,}")
    print(f"  Val (15%):   {val_size:,}")
    print(f"  Test (15%):  {test_size:,}")

    return labels


if __name__ == "__main__":
    labels = create_labels()
