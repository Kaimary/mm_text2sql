from dataset_readers import SpiderWikiDatasetReader
from allennlp.common.util import ensure_list
if __name__ =="__main__":
    #reader = SpiderWikiDatasetReader(tables_file='../data/spider_data/tables.json',dataset_path='C:\\Users\\bwy\\TODO\\nlp\\data_struct\\dataset\\database')
    #reader.read('../data/spider_data/dev.json')

    reader=SpiderWikiDatasetReader(tables_file='../data/wikisql_data/dev.tables.jsonl',is_spider=False)
    #reader.read('../data/wikisql_data/dev.jsonl')
    instances=ensure_list(reader.read('../data/wikisql_data/dev.jsonl'))
    ins=instances[0]
    print(type(ins))
    print(ins.fields)
    print(ins)
