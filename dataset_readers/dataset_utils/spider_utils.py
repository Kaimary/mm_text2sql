"""
utility functions for reading spider dataset
"""
import json
import os
from typing import List,Dict,Optional
from allennlp.common import JsonDict
from collections import defaultdict
from dataset_readers.dataset_utils.common import  TableColumn, Table



def read_spider_dataset_schema(schema_path: str) -> Dict[str, Dict[str, Table]]:
    schemas: Dict[str,Dict[str,Table]] = defaultdict(dict)
    dbs_json_blob = json.load(open(schema_path,"r"))
    for db in dbs_json_blob:
        db_id = db['db_id']

        column_id_to_table={}  # column_index 2 table_name
        column_id_to_column={} # column_index 2 table_column(instance)

        for index, (column,text,column_type) in enumerate(zip(db['column_names_original'],db['column_names'],db['column_types'])):
            table_id ,column_name = column  # column_name 数据库中的列名
            _, column_text = text           # column_text 规范化后的列名

            table_name = db['table_names_original'][table_id] # 该列在数据库中的对应的表的表名

            if table_name not in schemas[db_id]:
                table_text = db['table_names'][table_id]  # normalized table name
                schemas[db_id][table_name]=Table(table_name,table_text,[])

            if column_name == "*":
                continue

            is_primary_key = index in db['primary_keys']
            table_column = TableColumn(column_name , column_text , column_type, is_primary_key, None )
            schemas[db_id][table_name].columns.append(table_column)

            column_id_to_table[index]= table_name
            column_id_to_column[index]= table_column

        for (c1,c2) in db['foreign_keys']:      # c1 foreign , c2 primary
            foreign_key = column_id_to_table[c2] + ':' + column_id_to_column[c2].name   # 外键对应的 table_name + column_name
            column_id_to_column[c1].foreign_key = foreign_key


    return {**schemas}

