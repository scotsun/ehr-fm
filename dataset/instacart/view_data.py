#!/usr/bin/env python
"""快速查看 Instacart 处理好的数据（不依赖 pandas）"""

import duckdb
from pathlib import Path

project_root = Path(__file__).parent
parquet_dir = project_root / "data/instacart.parquet"

print("=" * 70)
print("Instacart 处理好的数据预览")
print("=" * 70)
print()

# 连接 DuckDB（只查询，不转 pandas）
conn = duckdb.connect()

print(f"✅ 数据路径: {parquet_dir}")

# 统计总行数
result = conn.execute(f"SELECT COUNT(*) FROM '{parquet_dir}/**/*.parquet'").fetchone()
total_rows = result[0]
print(f"✅ 总行数: {total_rows:,}")
print()

# 统计信息（直接在 SQL 中计算，避免读取大量数据）
print("📈 统计信息:")
stats = conn.execute(f"""
    SELECT 
        COUNT(DISTINCT user_id) as n_users,
        COUNT(DISTINCT order_id) as n_orders,
        COUNT(DISTINCT product_id) as n_products
    FROM '{parquet_dir}/**/*.parquet'
""").fetchone()
print(f"  - 用户数: {stats[0]:,}")
print(f"  - 订单数: {stats[1]:,}")
print(f"  - 商品种类: {stats[2]:,}")
print()

# 查看数据结构（schema）
print("📊 数据结构:")
schema = conn.execute(f"DESCRIBE SELECT * FROM '{parquet_dir}/**/*.parquet' LIMIT 1").fetchall()
for col in schema:
    print(f"  - {col[0]}: {col[1]}")
print()

# 查看前 10 行样本数据
print("👀 前 10 行数据样本:")
rows = conn.execute(f"SELECT * FROM '{parquet_dir}/**/*.parquet' LIMIT 10").fetchall()
cols = conn.execute(f"SELECT * FROM '{parquet_dir}/**/*.parquet' LIMIT 1").description
col_names = [desc[0] for desc in cols]

# 打印表头
print("  ".join(f"{name:>15}" for name in col_names))
print("-" * 70)
# 打印数据行
for row in rows:
    print("  ".join(f"{str(val):>15}" for val in row))
print()

# 显示一个用户的完整购物历史
print("🛒 用户 1 的购物历史（按时间顺序）:")
user_history = conn.execute(f"""
    SELECT order_number, product_id, reordered
    FROM '{parquet_dir}/**/*.parquet'
    WHERE user_id = 1
    ORDER BY order_number
    LIMIT 20
""").fetchall()

print(f"{'订单序号':>10}  {'商品ID':>10}  {'是否复购':>10}")
print("-" * 35)
for row in user_history:
    print(f"{row[0]:>10}  {row[1]:>10}  {row[2]:>10}")

conn.close()

