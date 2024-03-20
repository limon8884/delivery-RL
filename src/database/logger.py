from datetime import datetime

from src.database.classes import Event, TableName


class DatabaseLogger:
    def __init__(self, run_id: int) -> None:
        self.run_id = run_id
        self.reset()

    def reset(self) -> None:
        self.data: dict[str, list[tuple[int, datetime, str]]] = {
            TableName.COURIER_TABLE.value: [],
            TableName.CLAIM_TABLE.value: [],
            TableName.ORDER_TABLE.value: [],
        }

    def insert(self, table_name: TableName, item_id: int, dttm: datetime, event: Event):
        self.data[table_name.value].append((item_id, dttm, event.value))
