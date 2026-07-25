{{ config(materialized='table') }}

/*
    pre_experiment_covariates
    -------------------------
    One row per assigned developer with covariates measured STRICTLY
    before their assignment timestamp. This is the leakage guard for
    CUPED: a covariate that peeks past assignment can absorb treatment
    effect and bias the adjusted estimate, so every input here is
    filtered on event_at < assigned_at.

    Columns mirror what the analysis layer expects
    (src/cuped.py, notebooks/03_results.ipynb):
        account_age_days, repo_count, commit_streak,
        pre_experiment_score

    pre_experiment_score is a bounded 0-1 blend of profile depth and
    recent pre-period activity. The weights mirror the covariate
    construction in data/simulate_cohort.py — if you change one,
    change the other, or theta estimates in CUPED will quietly degrade.

    Grain: developer_id (unique).
*/

with assignments as (

    select * from {{ ref('experiment_assignments') }}
    where not is_contaminated

),

developers as (

    select * from {{ ref('stg_platform__developers') }}

),

pre_period_activity as (

    -- activity in the 90 days before assignment, never after
    select
        assignments.developer_id,
        count(distinct case
            when events.event_name = 'deploy_succeeded' then events.event_id
        end) as pre_deploy_count,
        count(distinct cast(events.event_at as date)) as pre_active_days
    from assignments
    left join {{ ref('stg_platform__events') }} as events
        on events.developer_id = assignments.developer_id
       and events.event_at <  assignments.assigned_at
       and events.event_at >= {{ dbt.dateadd('day', -90, 'assignments.assigned_at') }}
    group by 1

),

joined as (

    select
        assignments.developer_id,
        assignments.variant,
        assignments.assigned_at,

        {{ dbt.datediff('developers.account_created_at', 'assignments.assigned_at', 'day') }}
                                                     as account_age_days,
        coalesce(developers.repo_count, 0)           as repo_count,
        coalesce(developers.commit_streak, 0)        as commit_streak,
        coalesce(pre_period_activity.pre_deploy_count, 0) as pre_deploy_count,
        coalesce(pre_period_activity.pre_active_days, 0)  as pre_active_days
    from assignments
    left join developers
        on developers.developer_id = assignments.developer_id
    left join pre_period_activity
        on pre_period_activity.developer_id = assignments.developer_id

)

select
    developer_id,
    variant,
    assigned_at,
    account_age_days,
    repo_count,
    commit_streak,
    pre_deploy_count,
    pre_active_days,

    -- bounded 0-1 composite; saturating transforms keep whales from
    -- dominating and keep the score roughly linear in the outcome
      0.40 * ( ln(1 + repo_count) / ln(1 + 200.0) )
    + 0.30 * ( least(commit_streak, 60) / 60.0 )
    + 0.30 * ( least(pre_active_days, 90) / 90.0 )
                                                     as pre_experiment_score
from joined
