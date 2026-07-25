{{ config(materialized='table') }}

/*
    user_metrics
    ------------
    One row per assigned developer with every metric the readout needs.
    All windows are anchored on assigned_at, per user, so a developer
    assigned on day 20 of the experiment is measured on THEIR days
    0-30, not the experiment's.

    Primary metric
        retained_30d          — has an active service after 30 days,
                                operationalized as any deploy_succeeded
                                or service_heartbeat event in days
                                [30, 37) post-assignment.

    Secondary metrics
        time_to_first_deploy_hours
        services_created_30d

    Guardrail metrics
        onboarding_completed
        support_tickets_30d

    Maturity: retention is undefined until a user is 37 days past
    assignment. is_mature marks rows whose primary metric is valid;
    readouts must filter on it or they'll count immature users as
    churned and bias retention downward.

    Grain: developer_id (unique).
*/

with assignments as (

    select * from {{ ref('experiment_assignments') }}
    where not is_contaminated

),

events as (

    select * from {{ ref('stg_platform__events') }}

),

windowed as (

    select
        assignments.developer_id,
        assignments.variant,
        assignments.assigned_at,

        -- primary: active service signal in the day-30 measurement window
        max(case
            when events.event_name in ('deploy_succeeded', 'service_heartbeat')
             and events.event_at >= {{ dbt.dateadd('day', 30, 'assignments.assigned_at') }}
             and events.event_at <  {{ dbt.dateadd('day', 37, 'assignments.assigned_at') }}
            then 1 else 0
        end) as retained_30d,

        -- secondary
        min(case
            when events.event_name = 'deploy_succeeded'
             and events.event_at >= assignments.assigned_at
            then events.event_at
        end) as first_deploy_at,

        count(distinct case
            when events.event_name = 'service_created'
             and events.event_at >= assignments.assigned_at
             and events.event_at <  {{ dbt.dateadd('day', 30, 'assignments.assigned_at') }}
            then events.event_id
        end) as services_created_30d,

        -- guardrails
        max(case
            when events.event_name = 'onboarding_completed'
             and events.event_at >= assignments.assigned_at
            then 1 else 0
        end) as onboarding_completed,

        count(distinct case
            when events.event_name = 'support_ticket_opened'
             and events.event_at >= assignments.assigned_at
             and events.event_at <  {{ dbt.dateadd('day', 30, 'assignments.assigned_at') }}
            then events.event_id
        end) as support_tickets_30d

    from assignments
    left join events
        on events.developer_id = assignments.developer_id
    group by 1, 2, 3

)

select
    developer_id,
    variant,
    assigned_at,
    retained_30d,
    {{ dbt.datediff('assigned_at', 'first_deploy_at', 'hour') }} as time_to_first_deploy_hours,
    services_created_30d,
    onboarding_completed,
    support_tickets_30d,
    case
        when {{ dbt.dateadd('day', 37, 'assigned_at') }} <= {{ dbt.current_timestamp() }}
        then true else false
    end as is_mature
from windowed
