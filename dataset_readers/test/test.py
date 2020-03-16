from dataset_readers import SpiderWikiDatasetReader

if __name__ =="__main__":
    reader = SpiderWikiDatasetReader(tables_path='./../../data/spider_data/tables.json', isSpider=True)
    reader.read('../data/spider_data/dev.json')
