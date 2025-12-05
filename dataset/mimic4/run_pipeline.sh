#!/bin/bash
# MIMIC-IV Data Pipeline Runner
# Executes assets in correct dependency order

set -e  # Exit on error

cd "$(dirname "$0")"

# Load environment variables
export DATA_PROJECT_ROOT="$(pwd)"

echo "=========================================="
echo "MIMIC-IV Data Pipeline"
echo "=========================================="

# Stage 1: Raw data import
echo ""
echo "[Stage 1/6] Importing raw tables..."
uv run dg launch --assets "main/raw_admissions"
uv run dg launch --assets "main/raw_diagnoses_icd"
uv run dg launch --assets "main/raw_prescriptions"
uv run dg launch --assets "main/raw_procedures_icd"
uv run dg launch --assets "main/raw_labevents"
uv run dg launch --assets "main/raw_emar"

# Stage 2: Mapping tables
echo ""
echo "[Stage 2/6] Importing mapping tables..."
uv run dg launch --assets "main/mapping_gsn_atc"
uv run dg launch --assets "main/mapping_icd_cm_9_to_10"
uv run dg launch --assets "main/mapping_icd_pcs_9_to_10"

# Stage 3: Lookup tables (depend on raw + mapping)
echo ""
echo "[Stage 3/6] Building lookup tables..."
uv run dg launch --assets "main/drug_to_atc"
uv run dg launch --assets "main/icd_cm_9_to_10"
uv run dg launch --assets "main/icd_pcs_9_to_10"
uv run dg launch --assets "main/top_200_labs"

# Stage 4: Lab quantiles (depends on top_200_labs)
echo ""
echo "[Stage 4/6] Calculating lab quantiles (100% data, may take a while)..."
uv run dg launch --assets "main/lab_quantiles"

# Stage 5: dbt transformation
echo ""
echo "[Stage 5/6] Running dbt transformation..."
cd transform/nbr
uv run dbt run
cd ../..

# Stage 6: Export
echo ""
echo "[Stage 6/6] Exporting tokens to Parquet..."
uv run dg launch --assets "main/export_tokens"

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
