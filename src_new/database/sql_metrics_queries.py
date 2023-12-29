from src_new.database.classes import TableName, Event


def _sql_query_cr(run_id: int) -> str:
    return f'''
    select
        1.0 * sum(case when event = '{Event.CLAIM_COMPLETED.value}' then 1 else 0 end)
            / sum(case when event = '{Event.CLAIM_CREATED.value}' then 1 else 0 end)
                as cr
    from {TableName.CLAIM_TABLE.value}
    where run_id = {run_id}
    '''


def _sql_query_ctd(run_id) -> str:
    return f'''
    select avg(t.ctd)
    from (
        select
            claim_id,
            max(case when event = '{Event.CLAIM_COMPLETED.value}' then 1 else 0 end) as is_completed,
            (
                strftime('%s', max(case when event = '{Event.CLAIM_COMPLETED.value}' then dttm else null end)) 
                - strftime('%s', max(case when event = '{Event.CLAIM_CREATED.value}' then dttm else null end))
            ) as ctd
        from {TableName.CLAIM_TABLE.value}
        where run_id = {run_id}
        group by claim_id
    ) as t
    where t.is_completed = 1
    '''
