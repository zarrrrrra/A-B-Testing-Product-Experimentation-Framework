{{ config(materialized='view') }}

/*
    stg_platform__events
    --------------------
    Normalizes the raw event stream:
      * casts timestamps
      * lowercases and trims event names
      * drops internal/test accounts
      * dedupes on event_id (pipeline delivers at-least-once)

    One row per unique event. No business logic beyond normalization —
    metric definitions live in the marts layer.
*/

with source as (

    select * from {{ source('raw_platform', 'events') }}

),

renamed as (

    select
        cast(event_id as {{ dbt.type_string() }})            as event_id,
        cast(user_id as {{ dbt.type_string() }})             as developer_id,
        lower(trim(cast(event_name as {{ dbt.type_string() }}))) as event_name,
        cast(event_ts as {{ dbt.type_timestamp() }})         as event_at,
        properties
    from source
    where user_id is not null
      and event_ts is not null
      -- internal accounts and synthetic monitoring traffic
      and lower(cast(user_id as {{ dbt.type_string() }})) not like 'test\_%' escape '\\'

),

deduped as (

    select
        *,
        row_number() over (
            partition by event_id
            order by event_at
        ) as _rn
    from renamed

)

select
    event_id,
    developer_id,
    event_name,
    event_at,
    properties
from deduped
where _rn = 1
