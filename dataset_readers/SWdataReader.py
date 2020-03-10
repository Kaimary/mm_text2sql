import json
import logging
import os
from typing import List,Dict,Iterator
import tqdm

from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, Tokenizer, TokenIndexer, Field, Instance
from allennlp.data.fields import TextField, ProductionRuleField, ListField, IndexField, MetadataField,LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from overrides import overrides
from spacy.symbols import ORTH, LEMMA


logger=logging.getLogger(__name__)

@DatasetReader.register("swreader")
class SpiderWikiDatasetReader(DatasetReader):
    def __init__(self,
                 lazy:bool =False,
                 token_indexers: Dict[str,TokenIndexer] =None,
                 tokenizer: Tokenizer=None ,
                 tables_path: str =None,
                 isSpider: bool =True
                 ):
        """
        :param lazy:         lazy loading // to do
        :param token_indexers:
        :param tokenizer:     // spacyWordSplitter
        :param tables_path:   //schema path
        :param dataset_path:  // for
        :param isSpider:    is Spider or not?
        """
        super().__init__(lazy=lazy)

        # default spacy tokenizer splits the common token 'id' to ['i', 'd'], we here write a manual fix for that
        spacy_tokenizer= SpacyWordSplitter(pos_tags=True)
        spacy_tokenizer.spacy.tokenizer.add_special_case(u'id', [{ORTH: u'id', LEMMA: u'id'}])

        #setup
        self._tokenizer = tokenizer or WordTokenizer(spacy_tokenizer)
        self._utterance_token_indexers= token_indexers or {'tokens': SingleIdTokenIndexer()}

        self._tables_path = tables_path
        #self._dataset_path= dataset_path

        self._isSpider = isSpider

        if isSpider:
            if not tables_path.endswith('.json'):
                raise ConfigurationError(f"tables_path of Spider error...{tables_path}")
        else:
            if not tables_path.endswith('.jsonl'):
                raise ConfigurationError(f"tables_path of Wiki error...{tables_path}")


    @overrides
    def _read(self, file_path: str):
        if self._isSpider:
            if not file_path.endswith('.json'):
                raise ConfigurationError(f"dataset_path of Spider error...{file_path}")
            logger.info("reading instance from file at: %s",file_path)
            with open(file_path,"r") as data_file:
                json_objects=json.load(data_file)
                for _index,ex in enumerate(json_objects):
                    if 'query' in ex:
                        instance = self.text_to_instance(ex['question'],ex['db_id'],sql=ex['query'])
                    else:
                        instance = self.text_to_instance(ex['question'], ex['db_id'])
                    if instance is not None:
                        yield instance

        # for wiki
        else:
            if not file_path.endswith('.jsonl'):
                raise ConfigurationError(f"dataset_path of Wiki error...{file_path}")
            logger.info("reading instance from file at: %s",file_path)
            with open(file_path,"r") as data_file:
                for _index,line in enumerate(data_file.readlines()):
                    line=line.strip("\n")
                    if not line:
                        continue
                    ex = json.loads(line)
                    if 'sql' in ex:
                        instance=self.text_to_instance(ex['question'],ex['table_id'],sqldict=ex['sql'])
                    else:
                        instance=self.text_to_instance(ex['question'],ex['table_id'])
                    if instance is not None:
                        yield  instance

    @overrides
    def text_to_instance(self,
                         utterance: str,
                         db_id: str,
                         sql: str =None,
                         sqldict: Dict=None) -> Instance:

        fields: Dict[str,Field] = {}
        if self._isSpider:
            # question tokenizer...indexer...
            tokenized_utter= self._tokenizer.tokenize(utterance)
            utter_field = TextField(tokenized_utter, self._utterance_token_indexers)

            # db_info
            db_field=MetadataField(db_id)

            fields={"question":utter_field,"db":db_field}
            fields['isSpider']=MetadataField(True)

            if sql is not None:
                # optional
                tokenized_sql=self._tokenizer.tokenize(sql)
                sql_field=TextField(tokenized_sql,self._utterance_token_indexers)

                fields["sql"]=sql_field
        else:
            # question tokenizer...indexer...
            tokenized_utter = self._tokenizer.tokenize(utterance)
            utter_field = TextField(tokenized_utter, self._utterance_token_indexers)

            # db_info
            db_field = MetadataField(db_id)

            fields = {"question": utter_field, "db": db_field}
            fields['isSpider'] = MetadataField(False)

            if sqldict is not None:
                # optional
                sqldict_field = MetadataField(sqldict)

                fields["sqldict"]=sqldict_field


        return Instance(fields)




