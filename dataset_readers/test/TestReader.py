from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list
from allennlp.data.tokenizers import WordTokenizer

from dataset_readers import SpiderWikiDatasetReader
from dataset_readers import SpiderDBContext
from dataset_readers import WikiDBContext

import logging
# bunch of shit
logger=logging.getLogger(__name__)

class TestReader(AllenNlpTestCase):
    '''
    def test_read_spider_from_json(self):
        reader=SpiderWikiDatasetReader(tables_path='../data/spider_data/tables.json',isSpider=True)
        instances=ensure_list(reader.read('../data/spider_data/dev.json'))

        fields=instances[0].fields
        assert fields['db'].metadata == 'concert_singer'
        assert len(instances) == 1034 # dev.json length
        logger.warning("finish")   # dont work



    def test_read_wiki_from_json(self):
        reader = SpiderWikiDatasetReader(tables_path='../data/wikisql_data/dev.tables.jsonl', isSpider=False)
        instances = ensure_list(reader.read('../data/wikisql_data/dev.jsonl'))

        fields = instances[0].fields
        assert fields['db'].metadata == "1-10015132-11"
        assert len(instances) == 8421  # dev.jsonl length
    '''
    def test_spider_db_context(self):
        spiderDBcontext = SpiderDBContext(WordTokenizer(),'../../data/spider_data/tables.json')
        assert  len(spiderDBcontext.schemas) == 166
        assert  spiderDBcontext.schemas["perpetrator"]["people"].name == 'people'
        assert  len(spiderDBcontext.schemas["perpetrator"]["people"].columns) == 5
        assert  spiderDBcontext.schemas["perpetrator"]["people"].columns[0].name == 'People_ID'

    def test_wiki_db_context(self):
        wikicontext = WikiDBContext(WordTokenizer(),'../../data/wikisql_data/dev.tables.jsonl')
        assert  len(wikicontext.schemas) == 2716  # 2716 db
        assert  wikicontext.schemas["1-10015132-11"]["1-10015132-11"].name == "1-10015132-11"  # "1-10015132-11"
        assert  len(wikicontext.schemas["1-10015132-11"]["1-10015132-11"].columns) == 6  #6
        assert  wikicontext.schemas["1-10015132-11"]["1-10015132-11"].columns[0].name == 'col0'  # "col0"
        assert wikicontext.schemas["1-10015132-11"]["1-10015132-11"].columns[0].column_type == 'text'   # "text"
        assert wikicontext.schemas["1-10015132-11"]["1-10015132-11"].columns[0].text == 'Player'  # "text"