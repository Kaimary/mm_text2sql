import os
from typing import Dict,Tuple,List
from allennlp.data import Tokenizer,Token
from dataset_readers.dataset_utils.spider_utils import read_spider_dataset_schema
from dataset_readers.dataset_utils.wikisql_utils import read_wiki_dataset_schema

class DBContext:
    def __init__(self,tokenizer: Tokenizer, table_path: str):
        self._tokenizer = tokenizer
        self._table_path = table_path



class SpiderDBContext(DBContext):
    schemas={}
    def __init__(self,tokenizer: Tokenizer, table_path: str):
        super().__init__(tokenizer,table_path)
        SpiderDBContext.schemas=read_spider_dataset_schema(table_path)

        # todo





class WikiDBContext(DBContext):
    schemas={}
    def __init__(self,tokenizer:Tokenizer,table_path: str):
        super().__init__(tokenizer,table_path)
        WikiDBContext.schemas=read_wiki_dataset_schema(table_path)


        # todo


