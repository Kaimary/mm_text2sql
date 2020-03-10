"""
utility functions for reading wikisql dataset
"""

from typing import  Dict,List
from dataset_readers.dataset_utils.common import Table,TableColumn
from collections import  defaultdict
import json

def read_wiki_dataset_schema(schema_path: str) -> Dict[str, Dict[str, Table]]:
    schemas: Dict[str, Dict[str, Table]] = defaultdict(dict)
    with open(schema_path,"r") as df:
        for _,line in enumerate(df.readlines()):
            line = line.strip("\n")
            if not line:
                continue
            db = json.loads(line)

            db_id =db['id']

            # 暂时使用db_id 替代table_name和table_text
            # table_name = db['name']  # table_name in .db file     1%左右 部分json对象中是没有 'name' 字段的 我真的是无语了
            schemas[db_id][db_id]=Table(db_id,db_id,[]) # table_text 在wikisql 中不存在，先用db_id 填充
            for index,(column_text,column_type) in enumerate(zip(db['header'],db['types'])):

                if column_text == "*":
                    continue
                # 在wikisql 中 真正的column_name 是 [col0,col1]
                # 而且也没有主键和外键的信息，所以这里填入 False , None
                table_column =TableColumn('col'+str(index),column_text,column_type,False,None)
                schemas[db_id][db_id].columns.append(table_column)
    # todo  finish reading
    return {**schemas}
