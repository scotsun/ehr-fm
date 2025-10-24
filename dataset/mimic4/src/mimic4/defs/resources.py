import dagster as dg
from dagster_duckdb import DuckDBResource

from ..constants import DATA_PROJECT_ROOT

duckdb_resource = DuckDBResource(
    database=str(DATA_PROJECT_ROOT / "data/mimic4.duckdb"),
)


@dg.definitions
def resources() -> dg.Definitions:
    return dg.Definitions(
        resources={
            "duckdb": duckdb_resource,
        }
    )

