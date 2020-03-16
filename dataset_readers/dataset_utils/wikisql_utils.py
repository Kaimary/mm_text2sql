"""
utility functions for reading wikisql dataset
"""
from allennlp.common import  JsonDict
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

wiki_schemas = None
def gen_wiki_tokens(ex: JsonDict,schema_path: str):
    """

    :param ex:
    :param schema_path:
    :return:
    ['select',
     'avg', '(', 'singer@age', ')', ',',
     'min', '(', 'singer@age', ')', ',',
     'max', '(', 'singer@age', ')',
     'from', 'singer',
     'where', 'singer@country', '=', "'value'"]
     spider 的输出就是这样的


     我也对列名做了 lower()的处理


    """
    def get_schema(schema_path:str):
        global wiki_schemas
        if wiki_schemas is not None:

            return wiki_schemas
        else:

            wiki_schemas=read_wiki_dataset_schema(schema_path)
            return wiki_schemas


    schemas=get_schema(schema_path)
    #print(len(schemas))
    agg_ops = ['', 'max', 'min', 'count', 'sum', 'avg']
    cond_ops = ['=', '>', '<', 'OP']

    query_tokens=[]

    sel = ex['sql']['sel']
    conds = ex['sql']['conds']
    agg = ex['sql']['agg']
    table_id = ex['table_id']
    #print(table_id)
    query_tokens.append('select')
    if agg != 0 :
        query_tokens.append(agg_ops[agg])
        query_tokens.append('(')
        query_tokens.append('{table_id}@{col_name}'.format(table_id=table_id,col_name=schemas[table_id][table_id].columns[sel].text.lower()))
        query_tokens.append(')')
    else :
        query_tokens.append('{table_id}@{col_name}'.format(table_id=table_id,col_name=schemas[table_id][table_id].columns[sel].text.lower()))
    query_tokens.append('from')
    query_tokens.append('{table_id}'.format(table_id=table_id))

    if len(conds) >0:
        query_tokens.append('where')
        for index, cond in enumerate(conds):
            if index>0:
                query_tokens.append('and')
            query_tokens.append('{table_id}@{col_name}'.format(table_id=table_id,
                                                               col_name=schemas[table_id][table_id].columns[
                                                                   cond[0]].text.lower()))
            query_tokens.append(cond_ops[cond[1]])
            query_tokens.append("'value'")


    return query_tokens

schemas_value=None
def read_wiki_dataset_values(db_id : str, table_path: str , tables: List[str]):
    """

    :param db_id:
    :param table_path:
    :param tables:
    :return: {<dataset_readers.dataset_utils.common.Table object at 0x0000022329E57288>:
    [(1, '2'), (1, '3'), (1, '5'), (2, '3'), (2, '6'), (3, '5'), (4, '4'), (5, '6'), (5, '3'), (6, '2')],
    <dataset_readers.dataset_utils.common.Table object at 0x0000022329E57308>:
    [(1, 'Raith Rovers', "Stark's Park", 10104, 4812, 1294, 2106), (2, 'Ayr United', 'Somerset Park'
    , 11998, 2363, 1057, 1477), (3, 'East Fife', 'Bayview Stadium', 2000, 1980, 533, 864), (4, "Queen's Park"
    , 'Hampden Park', 52500, 1763, 466, 730), (5, 'Stirling Albion', 'Forthbank Stadium', 3808, 1125, 404,
    642), (6, 'Arbroath', 'Gayfield Park', 4125, 921, 411, 638), (7, 'Alloa Athletic', 'Recreation Park',
     3100, 1057, 331, 637), (9, 'Peterhead', 'Balmoor', 4000, 837, 400, 615), (10, 'Brechin City', 'Glebe Par
     k', 3960, 780, 315, 552)],  'F')
    """
    def get_schema_value(schema_path:str):
        global schemas_value
        if schemas_value is not None:
            #print("yes")
            return schemas_value
        else:
            #print("no")
            schemas_value={}
            with open(schema_path, "r") as df:
                for _, line in enumerate(df.readlines()):
                    line = line.strip("\n")
                    if not line:
                        continue
                    db = json.loads(line)

                    db_id = db['id']
                    rows=[tuple(row) for row in db['rows']]
                    #print(rows)
                    schemas_value[db_id]=rows
            # todo  finish reading
            return schemas_value


    schemas=get_schema_value(table_path)

    values={}
    for table in tables:
        values[table] = schemas[table.name]
    return values
