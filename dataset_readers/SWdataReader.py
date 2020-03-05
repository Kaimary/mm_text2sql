import json
import logging
import os
from typing import List,Dict,Iterator
import tqdm

from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, Tokenizer, TokenIndexer, Field, Instance
from allennlp.data.fields import TextField, ProductionRuleField, ListField, IndexField, MetadataField
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
        if self._isSpider :
            if not file_path.endswith('.json'):
                raise ConfigurationError(f"dataset_path of Spider error...{file_path}")
            logger.info("reading instance from file at: %s",file_path)
        else:
            if not file_path.endswith('.jsonl'):
                raise ConfigurationError(f"dataset_path of Wiki error...{file_path}")
            logger.info("reading instance from file at: %s",file_path)

    @overrides
    def text_to_instance(self, *inputs) -> Instance:
        fields: Dict[str,Field] = {}
        return Instance(fields)




