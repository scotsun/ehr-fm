import mlflow
from mlflow.tracking import MlflowClient
import time
import os
import yaml

# CONFIGURATION
# 1. The root of your local mlruns folder (e.g., current directory)
MLRUNS_ROOT = os.path.abspath("./mlruns")

# 2. The old prefix to look for (from your error message)
OLD_PREFIX = "/hpc/home/ms1008/ehr-fm/mlruns"
# Note: Sometimes it might just be "/hpc/..." without "file://".
# Check one of your meta.yaml files if this script doesn't catch it.


def fix_meta_files(root_dir):
    print(f"Scanning {root_dir}...")

    # Iterate through all files in mlruns
    for dirpath, _, filenames in os.walk(root_dir):
        if "meta.yaml" in filenames:
            file_path = os.path.join(dirpath, "meta.yaml")

            with open(file_path, "r") as f:
                content = f.read()

            # Check if the file contains the old prefix
            if OLD_PREFIX in content:
                # Create the new valid local path
                # We need to construct the correct path dynamically
                # New format: file:///abs/path/to/mlruns/EXP_ID/RUN_ID/artifacts

                # Determine relative path from mlruns root to this folder
                rel_path = os.path.relpath(dirpath, root_dir)

                # Construct new absolute prefix
                new_prefix = f"file://{MLRUNS_ROOT}"

                # Perform replacement
                new_content = content.replace(OLD_PREFIX, new_prefix)

                with open(file_path, "w") as f:
                    f.write(new_content)

                print(f"Fixed: {rel_path}/meta.yaml")


fix_meta_files(MLRUNS_ROOT)
print("Done. You can now open `mlflow ui` or run the migration scripts.")


# Configuration
SOURCE_EXP_ID = "748975335134678951"
TARGET_EXP_ID = "428834719722401669"

client = MlflowClient()

# Get all runs from source
runs = client.search_runs(SOURCE_EXP_ID)

for run in runs:
    print(f"Migrating Run: {run.info.run_id} (only metrics not artifact)")

    # 1. Create new run with same start time and tags
    new_run = client.create_run(
        experiment_id=TARGET_EXP_ID,
        tags=run.data.tags,
        start_time=run.info.start_time,
        run_name=run.data.tags.get("mlflow.runName"),
    )

    # 2. Log Parameters (Single values)
    for key, val in run.data.params.items():
        client.log_param(new_run.info.run_id, key, val)

    # 3. Log Metrics (FULL HISTORY)
    for metric_key in run.data.metrics.keys():
        # Fetch full history of the metric
        history = client.get_metric_history(run.info.run_id, metric_key)

        # Log each data point with its specific timestamp and step
        # Note: log_batch is faster but has a limit of 1000 items per call.
        # For simplicity, we loop here, but for massive history, batching is better.
        for m in history:
            client.log_metric(
                run_id=new_run.info.run_id,
                key=m.key,
                value=m.value,
                timestamp=m.timestamp,
                step=m.step,
            )

    print(f"Done! New Run ID: {new_run.info.run_id}")
