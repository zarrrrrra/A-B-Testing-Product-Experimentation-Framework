{{ config(materialized='view') }}

/*
    stg_platform__exposures
    -----------------------
    Experiment exposure events, normalized. A user can be exposed many
    times (every onboarding page view fires one); analysis cares about
    the FIRST exposure, which defines assignment time.

    We also count distinct variants seen per (developer, experiment) so
    the assignments mart can quarantine contaminated users — anyone who
    somehow saw both arms.
*/

with source as (

    select * from {{ source('raw_platform', 'experiment_exposures') }}

),

renamed as (

    select
        cast(exposure_id as {{ dbt.type_string() }})   as exposure_id,
        cast(user_id as {{ dbt.type_string() }})       as developer_id,
        cast(experiment_id as {{ dbt.type_string() }}) as experiment_id,
        lower(trim(cast(variant as {{ dbt.type_string() }}))) as variant,
        cast(exposed_at as {{ dbt.type_timestamp() }}) as exposed_at
    from source
    where user_id is not null
      and variant in ('control', 'treatment')

),

first_exposure as (

    select
        *,
        row_number() over (
            partition by developer_id, experiment_id
            order by exposed_at, exposure_id
        ) as _rn,
        count(distinct variant) over (
            partition by developer_id, experiment_id
        ) as variants_seen
    from renamed

)

select
    exposure_id,
    developer_id,
    experiment_id,
    variant,
    exposed_at,
    variants_seen
from first_exposure
where _rn = 1
