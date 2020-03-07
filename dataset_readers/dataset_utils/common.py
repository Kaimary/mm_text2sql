from typing import  Optional,List

class TableColumn:
    def __init__(self,
                 name: str,
                 text: str,
                 column_type: str,
                 is_primary_key: bool,
                 foreign_key: Optional[str]):
        self.name=name
        self.text=text
        self.column_type=column_type
        self.is_primary_key=is_primary_key
        self.foreign_key = foreign_key

class Table:
    def __init__(self, name: str , text: str  , columns: List[TableColumn] ):
        self.name= name
        self.text= text
        self.columns = columns
