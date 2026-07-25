{{ config(materialized='view') }}

/*
    stg_platform__developers
    ------------------------
    Developer profile attributes sourced from the GitHub public API
    (data/github_fetch.py). A developer can be fetched more than once;
    we keep the most recent snapshot per developer.

    Age-style covariates (e.g. account_age_days) are deliberately NOT
    computed here: they must be measured relative to the experiment
    assignment timestamp to avoid leakage, so they live in
    marts/pre_experiment_covariates.
*/

with source as (

    select * from {{ source('raw_platform', 'developers') }}

),

renamed as (

    select
        cast(user_id as {{ dbt.type_string() }})             as developer_id,
        cast(github_login as {{ dbt.type_string() }})        as github_login,
        cast(account_created_at as {{ dbt.type_timestamp() }}) as account_created_at,
        cast(public_repo_count as {{ dbt.type_int() }})      as repo_count,
        cast(top_language as {{ dbt.type_string() }})        as top_language,
        cast(commit_streak_days as {{ dbt.type_int() }})     as commit_streak,
        cast(fetched_at as {{ dbt.type_timestamp() }})       as fetched_at
    from source
    where user_id is not null

),

latest_snapshot as (

    select
        *,
        row_number() over (
            partition by developer_id
            order by fetched_at desc
        ) as _rn
    from renamed

)

select
    developer_id,
    github_login,
    account_created_at,
    repo_count,
    top_language,
    commit_streak,
    fetched_at
from latest_snapshot
where _rn = 1
