with raw_orders as (
    select * from {{ source('main', 'raw_orders') }}
),
raw_order_products as (
    select * from {{ source('main', 'raw_order_products') }}
)

select
    o.user_id,
    o.order_id,
    o.order_number,
    o.days_since_prior_order,
    op.product_id,
    op.reordered,
from raw_orders o
join raw_order_products op 
on o.order_id = op.order_id