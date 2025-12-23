"""
Create downstream task labels from MIMIC-IV raw data.

Tasks: mortality, readmission_30d, prolonged_los, icd_chapter, icd_category_multilabel
Output: downstream_labels.csv + icd_category_vocab.json
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import json

# Paths
RAW_DIR = Path(__file__).parent / "data" / "raw_mimic4"
OUTPUT_DIR = Path(__file__).parent / "data"
PARQUET_DIR = OUTPUT_DIR / "mimic4_tokens.parquet"


def get_icd_chapter(icd_code: str, version: int) -> int:
    """Map ICD code to chapter number (1-22 for ICD-10, 1-21 for ICD-9). Returns -1 if unknown."""
    if pd.isna(icd_code):
        return -1

    code = str(icd_code).strip().upper()

    if version == 10:
        first_char = code[0] if code else ''

        # D codes span two chapters
        if first_char == 'D':
            try:
                numeric_part = ''.join(c for c in code[1:] if c.isdigit())
                if numeric_part:
                    num = int(numeric_part[:2])
                    if num < 50:
                        return 2   # D00-D49: Neoplasms
                    else:
                        return 3   # D50-D89: Blood diseases
            except (ValueError, IndexError):
                pass
            return 2

        # H codes span two chapters
        if first_char == 'H':
            try:
                numeric_part = ''.join(c for c in code[1:] if c.isdigit())
                if numeric_part:
                    num = int(numeric_part[:2])
                    if num < 60:
                        return 7   # H00-H59: Eye
                    else:
                        return 8   # H60-H95: Ear
            except (ValueError, IndexError):
                pass
            return 7

        chapter_map = {
            'A': 1, 'B': 1, 'C': 2, 'E': 4, 'F': 5, 'G': 6,
            'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14,
            'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 19,
            'V': 20, 'W': 20, 'X': 20, 'Y': 20, 'Z': 21, 'U': 22,
        }
        return chapter_map.get(first_char, -1)

    elif version == 9:
        try:
            numeric = ''.join(c for c in code if c.isdigit() or c == '.')
            if not numeric:
                if code.startswith('E'):
                    return 19
                elif code.startswith('V'):
                    return 21
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


def get_icd_category(icd_code: str, version: int) -> str:
    """Get 3-char ICD category. ICD-10: first 3 chars; ICD-9: '9_' + first 3 digits."""
    if pd.isna(icd_code):
        return None
    code = str(icd_code).strip().upper()
    if len(code) < 3:
        return None
    if version == 10:
        return code[:3]
    elif version == 9:
        return f"9_{code[:3]}"
    return None


def create_icd_category_labels(diagnoses: pd.DataFrame, valid_hadm_ids: set) -> dict:
    """Create multi-label ICD category data for each admission."""
    print("\n[Extra] Creating ICD Category multi-label data...")

    valid_dx = diagnoses[diagnoses['hadm_id'].isin(valid_hadm_ids)].copy()
    valid_dx['icd_category'] = valid_dx.apply(
        lambda x: get_icd_category(x['icd_code'], x['icd_version']), axis=1
    )
    valid_dx = valid_dx[valid_dx['icd_category'].notna()]

    category_counts = valid_dx['icd_category'].value_counts()
    min_count = 100
    valid_categories = category_counts[category_counts >= min_count].index.tolist()
    print(f"  Total unique categories: {len(category_counts)}")
    print(f"  Categories with >= {min_count} occurrences: {len(valid_categories)}")

    category_to_idx = {cat: idx for idx, cat in enumerate(sorted(valid_categories))}
    idx_to_category = {idx: cat for cat, idx in category_to_idx.items()}
    valid_dx = valid_dx[valid_dx['icd_category'].isin(valid_categories)]

    hadm_to_categories = {}
    for hadm_id, group in valid_dx.groupby('hadm_id'):
        categories = group['icd_category'].unique().tolist()
        category_indices = [category_to_idx[cat] for cat in categories if cat in category_to_idx]
        if category_indices:
            hadm_to_categories[int(hadm_id)] = category_indices

    num_labels_per_admission = [len(cats) for cats in hadm_to_categories.values()]
    print(f"  Admissions with valid categories: {len(hadm_to_categories):,}")
    print(f"  Avg categories per admission: {np.mean(num_labels_per_admission):.1f}")
    print(f"  Max categories per admission: {max(num_labels_per_admission)}")

    return {
        'hadm_to_categories': hadm_to_categories,
        'category_to_idx': category_to_idx,
        'idx_to_category': idx_to_category,
        'num_categories': len(valid_categories),
    }


def create_labels():
    print("=" * 60)
    print("Creating downstream task labels from MIMIC-IV")
    print("=" * 60)

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

    parquet_patients = set()
    for d in PARQUET_DIR.glob("subject_id=*"):
        parquet_patients.add(int(d.name.split('=')[1]))
    print(f"  Parquet patients: {len(parquet_patients):,}")

    admissions = admissions[admissions['subject_id'].isin(parquet_patients)]
    print(f"  Filtered admissions: {len(admissions):,}")

    # Task 1: Inpatient Mortality
    print("\n[2/6] Creating mortality labels...")
    admissions['mortality'] = admissions['hospital_expire_flag'].astype(int)
    mortality_rate = admissions['mortality'].mean()
    print(f"  Mortality rate: {mortality_rate:.2%} ({admissions['mortality'].sum():,} / {len(admissions):,})")

    print("\n[3/6] Creating 30-day readmission labels...")
    admissions = admissions.sort_values(['subject_id', 'admittime'])
    admissions['next_admittime'] = admissions.groupby('subject_id')['admittime'].shift(-1)
    admissions['days_to_next'] = (admissions['next_admittime'] - admissions['dischtime']).dt.total_seconds() / 86400

    admissions['readmission_30d'] = (
        (admissions['days_to_next'] <= 30) &
        (admissions['days_to_next'] >= 0) &
        (admissions['mortality'] == 0)
    ).astype(int)

    admissions.loc[admissions['mortality'] == 1, 'readmission_30d'] = -1
    admissions.loc[admissions['next_admittime'].isna(), 'readmission_30d'] = -1

    eligible = admissions[admissions['readmission_30d'] >= 0]
    readmission_rate = eligible['readmission_30d'].mean()
    print(f"  Readmission rate: {readmission_rate:.2%} ({eligible['readmission_30d'].sum():,} / {len(eligible):,})")
    print(f"  Ineligible (death/last admission): {(admissions['readmission_30d'] == -1).sum():,}")

    print("\n[4/6] Creating prolonged LoS labels...")
    admissions['los_days'] = (admissions['dischtime'] - admissions['admittime']).dt.total_seconds() / 86400
    admissions['prolonged_los'] = (admissions['los_days'] > 7).astype(int)

    prolonged_rate = admissions['prolonged_los'].mean()
    print(f"  Prolonged LoS rate: {prolonged_rate:.2%} ({admissions['prolonged_los'].sum():,} / {len(admissions):,})")
    print(f"  Mean LoS: {admissions['los_days'].mean():.1f} days, Median: {admissions['los_days'].median():.1f} days")

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

    print("\n[6/6] Creating ICD category multi-label data...")
    valid_hadm_ids = set(admissions['hadm_id'].tolist())
    icd_category_data = create_icd_category_labels(diagnoses, valid_hadm_ids)

    print("\n" + "=" * 60)
    print("Saving labels...")

    labels = admissions[[
        'subject_id', 'hadm_id', 'admittime',
        'mortality', 'readmission_30d', 'prolonged_los', 'icd_chapter'
    ]].copy()

    hadm_to_categories = icd_category_data['hadm_to_categories']
    labels['icd_categories'] = labels['hadm_id'].apply(
        lambda x: ','.join(map(str, hadm_to_categories.get(x, []))) if x in hadm_to_categories else ''
    )

    output_path = OUTPUT_DIR / "downstream_labels.csv"
    labels.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    print(f"  Total rows: {len(labels):,}")

    icd_category_path = OUTPUT_DIR / "icd_category_vocab.json"
    with open(icd_category_path, 'w') as f:
        save_data = {
            'category_to_idx': icd_category_data['category_to_idx'],
            'idx_to_category': {str(k): v for k, v in icd_category_data['idx_to_category'].items()},
            'num_categories': icd_category_data['num_categories'],
        }
        json.dump(save_data, f)
    print(f"  Saved ICD category vocab to: {icd_category_path}")
    print(f"  Number of categories: {icd_category_data['num_categories']}")

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
    print(f"    Unique chapters: {labels['icd_chapter'].nunique() - 1}")

    print("\n" + "=" * 60)
    print("Patient-Level Split:")
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
