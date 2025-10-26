import dagster as dg

from dagster_duckdb import DuckDBResource
from instacart.constants import DATA_PROJECT_ROOT, EXPORT_DIR


@dg.asset(kinds={"duckdb"}, key=["main", "raw_orders"])
def raw_orders(duckdb: DuckDBResource) -> None:
    """
    Import the orders.csv file to the raw_orders table.
    """
    with duckdb.get_connection() as conn:
        conn.execute(
            """
            create or replace table raw_orders as (
                select * from read_csv_auto('{filepath}')
            )
            """.format(
                filepath=str(DATA_PROJECT_ROOT / "data/raw_instacart/orders.csv")
            )
        )
    return


@dg.asset(kinds={"duckdb"}, key=["main", "raw_order_products"])
def raw_order_products(duckdb: DuckDBResource) -> None:
    """
    Import the order_products__prior.csv and order_products__train.csv files to the raw_order_products table.
    """
    with duckdb.get_connection() as conn:
        conn.execute(
            """
            create or replace table raw_order_products as (
                select * from read_csv_auto('{prior}')
                union all
                select * from read_csv_auto('{train}')
            )
            """.format(
                prior=str(
                    DATA_PROJECT_ROOT / "data/raw_instacart/order_products__prior.csv"
                ),
                train=str(
                    DATA_PROJECT_ROOT / "data/raw_instacart/order_products__train.csv"
                ),
            )
        )
    return


@dg.asset(
    kinds={"duckdb"},
    key=["main", "user"],
    deps=[
        dg.AssetKey(["main", "stg"]),
    ],
)
def user(duckdb: DuckDBResource) -> None:
    """
    Export the stg table to parquet files partitioned by user_id.
    """
    with duckdb.get_connection() as conn:
        conn.execute("SET preserve_insertion_order = false;")
        conn.execute(
            f"""
            COPY (
                select *
                from stg
                order by user_id, order_number
            )
            TO '{str(EXPORT_DIR)}'
            (
                format parquet,
                partition_by user_id,
                write_partition_columns true,
                overwrite_or_ignore true
            )
            """
        )
    return
