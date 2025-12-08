#!/usr/bin/env python3
"""
View processed MIMIC-IV data
Supports:
1. View first N patients
2. View specific patient by ID
"""

import sys
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

# Configuration
PARQUET_DIR = Path(__file__).parent / "data/mimic4_tokens.parquet"

def get_all_patients():
    """Get all patient ID list"""
    if not PARQUET_DIR.exists():
        print(f"❌ Data directory does not exist: {PARQUET_DIR}")
        return []
    
    patient_dirs = sorted(PARQUET_DIR.glob("subject_id=*"))
    patient_ids = [int(d.name.split("=")[1]) for d in patient_dirs]
    return sorted(patient_ids)

def load_patient_data(subject_id):
    """Load single patient data"""
    patient_dir = PARQUET_DIR / f"subject_id={subject_id}"
    parquet_file = patient_dir / "data_0.parquet"

    if not parquet_file.exists():
        print(f"❌ Patient {subject_id} data does not exist")
        return None

    # Use ParquetFile to avoid schema inference issues
    pf = pq.ParquetFile(parquet_file)
    df = pf.read().to_pandas()
    return df

def display_patient_summary(df, subject_id):
    """Display patient data summary"""
    print("=" * 80)
    print(f"Patient ID: {subject_id}")
    print("=" * 80)
    
    # Basic statistics
    n_admissions = df['hadm_id'].nunique()
    n_events = len(df)
    
    print(f"\n📊 Basic Information:")
    print(f"  - Number of admissions: {n_admissions}")
    print(f"  - Total events: {n_events}")
    print(f"  - Average per admission: {n_events/n_admissions:.1f} events")
    
    # Event type distribution
    print(f"\n📈 Event Type Distribution:")
    code_type_dist = df['code_type'].value_counts()
    for code_type, count in code_type_dist.items():
        pct = count / n_events * 100
        print(f"  - {code_type}: {count:,} ({pct:.1f}%)")
    
    # Admission history
    print(f"\n🏥 Admission History:")
    visits = df.groupby('hadm_id').agg({
        'visit_seq': 'first',
        'admittime': 'first',
        'days_since_prior_admission': 'first',
        'code': 'count'
    }).sort_values('visit_seq')
    visits.columns = ['visit_seq', 'admittime', 'days_since_prior', 'n_events']
    
    print(f"{'Visit':>6} {'Admission Time':>20} {'Days Since Prior':>16} {'Events':>8}")
    print("-" * 55)
    for idx, row in visits.iterrows():
        days_str = f"{row['days_since_prior']:.0f}" if pd.notna(row['days_since_prior']) else "N/A"
        print(f"{int(row['visit_seq']):>6} {str(row['admittime'])[:19]:>20} {days_str:>16} {int(row['n_events']):>8}")
    
    # Check LAB binning (format: LAB:{itemid}_Q{1-10})
    lab_codes = df[df['code_type'] == 'lab']['code'] if 'lab' in df['code_type'].values else pd.Series()
    has_lab_binning = lab_codes.str.contains('_Q\\d+').any() if len(lab_codes) > 0 else False

    print(f"\n✅ Data Quality Check:")
    print(f"  - LAB quantile binning: {'✓' if has_lab_binning else '✗'}")
    
    return df

def display_patient_events(df, subject_id, max_events=50):
    """Display detailed event sequence for patient"""
    print(f"\n📋 Event Sequence (first {max_events}):")
    print(f"{'Visit':>6} {'Code':>30} {'Type':>12} {'Time Offset':>12} {'Event Time':>20}")
    print("-" * 85)
    
    for idx, row in df.head(max_events).iterrows():
        time_offset = f"{row['time_offset_hours']:.1f}h" if pd.notna(row['time_offset_hours']) else "N/A"
        event_time = str(row['event_time'])[:19] if pd.notna(row['event_time']) else "N/A"
        code_display = str(row['code'])[:30]
        
        print(f"{int(row['visit_seq']):>6} {code_display:>30} {row['code_type']:>12} {time_offset:>12} {event_time:>20}")

def view_top_patients(n=5):
    """View first N patients"""
    all_patients = get_all_patients()
    
    if not all_patients:
        return
    
    print(f"\n🔍 Viewing first {n} patients")
    print(f"Total patients: {len(all_patients):,}\n")
    
    for i, patient_id in enumerate(all_patients[:n], 1):
        df = load_patient_data(patient_id)
        if df is not None:
            display_patient_summary(df, patient_id)
            display_patient_events(df, patient_id, max_events=20)
            if i < n:
                print("\n" + "─" * 80 + "\n")

def view_specific_patient(subject_id):
    """View specific patient"""
    print(f"\n🔍 Viewing patient {subject_id}")
    
    df = load_patient_data(subject_id)
    if df is not None:
        display_patient_summary(df, subject_id)
        display_patient_events(df, subject_id, max_events=100)
        
        # Display complete DataFrame info
        print(f"\n📊 Complete DataFrame Info:")
        print(df.info())
        
        # Return DataFrame for interactive use
        return df

def main():
    """Main function"""
    if not PARQUET_DIR.exists():
        print(f"❌ Data directory does not exist: {PARQUET_DIR}")
        print("Please run data processing pipeline first")
        return
    
    print("=" * 80)
    print("MIMIC-IV Data Viewer")
    print("=" * 80)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        try:
            subject_id = int(sys.argv[1])
            view_specific_patient(subject_id)
        except ValueError:
            print(f"❌ Invalid patient ID: {sys.argv[1]}")
            print("Usage: python view_data.py [patient_id]")
    else:
        # Display usage instructions
        all_patients = get_all_patients()
        print(f"\n📊 Data Overview:")
        print(f"  - Data path: {PARQUET_DIR}")
        print(f"  - Total patients: {len(all_patients):,}")
        if all_patients:
            print(f"  - Patient ID range: {min(all_patients)} - {max(all_patients)}")
        
        print(f"\n💡 Usage:")
        print(f"  1. View first 5 patients: python view_data.py")
        print(f"  2. View specific patient: python view_data.py 10000032")
        
        # Default: show first 5 patients
        view_top_patients(n=5)

if __name__ == "__main__":
    main()
