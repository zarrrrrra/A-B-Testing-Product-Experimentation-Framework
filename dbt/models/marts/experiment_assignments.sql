{{ config(materialized='table') }}

/*
    experiment_assignments
    ----------------------
    One row per developer enrolled in the autoscaling-tip onboarding
    experiment. Assignment time = first exposure time.

    Rules encoded here (and only here — downstream models just join):
      1. First exposure wins. Later exposures never change the arm.
      2. Users who saw BOTH variants are flagged is_contaminated and
         must be excluded from readouts (kept in the table so we can
         monitor the contamination rate as a data-quality signal).
      3. The experiment window is bounded by vars so re-running the
         project against a longer event history doesn't silently
         enroll out-of-window users.

    Grain: developer_id (unique).
*/

{% set experiment_id = var('experiment_id', 'autoscaling_tip_onboarding_v1') %}

with exposures as (

    select * from {{ ref('stg_platform__exposures') }}
    where experiment_id = '{{ experiment_id }}'

),

signups as (

    -- enrollment eligibility: the unit is a developer account that
    -- completed signup; one row per person, not per project/service
    select
        developer_id,
        min(event_at) as signed_up_at
    from {{ ref('stg_platform__events') }}
    where event_name = 'signup_completed'
    group by 1

)

select
    exposures.developer_id,
    exposures.experiment_id,
    exposures.variant,
    exposures.exposed_at                          as assigned_at,
    signups.signed_up_at,
    (exposures.variants_seen > 1)                 as is_contaminated
from exposures
inner join signups
    on signups.developer_id = exposures.developer_id
where exposures.exposed_at >= cast('{{ var("experiment_start", "2026-05-01") }}' as {{ dbt.type_timestamp() }})
  and exposures.exposed_at <  cast('{{ var("experiment_end",   "2026-05-29") }}' as {{ dbt.type_timestamp() }})
