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
from semparse.contexts.spider_db_context import SpiderDBContext
from semparse.contexts.wiki_db_context import WikiDBContext
from dataset_readers.dataset_utils import fix_number_value,disambiguate_items,gen_wiki_tokens
from dataset_readers.fields import SpiderKnowledgeGraphField,WikiKnowledgeGraphField
from semparse.worlds.spider_world import SpiderWorld
from semparse.worlds.wiki_world import WikiWorld


logger=logging.getLogger(__name__)

@DatasetReader.register("swreader")
class SpiderWikiDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 question_token_indexers: Dict[str, TokenIndexer] = None,
                 keep_if_unparsable: bool = True,
                 tables_file: str = None,
                 dataset_path: str = 'dataset/database',
                 load_cache: bool = True,
                 save_cache: bool = True,
                 loading_limit=-1,
                 is_spider: bool = True
                 ):
        super().__init__(lazy=lazy)
        # default spacy tokenizer splits the common token 'id' to ['i', 'd'], we here write a manual fix for that
        spacy_tokenizer = SpacyWordSplitter(pos_tags=True)
        spacy_tokenizer.spacy.tokenizer.add_special_case(u'id', [{ORTH: u'id', LEMMA: u'id'}])
        self._tokenizer = WordTokenizer(spacy_tokenizer)

        self._utterance_token_indexers = question_token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._keep_if_unparsable = keep_if_unparsable

        self._tables_file = tables_file
        self._dataset_path = dataset_path

        self._load_cache = load_cache
        self._save_cache = save_cache
        self._loading_limit = loading_limit

        self._is_spider = is_spider




    @overrides
    def _read(self, file_path: str):
        if self._is_spider:
            if not file_path.endswith('.json'):
                raise ConfigurationError(f"Don't know how to read filetype of {file_path}")

            with open(file_path, "r") as data_file:
                json_obj = json.load(data_file)
                for total_cnt, ex in enumerate(json_obj):

                    query_tokens = None
                    if 'query_toks' in ex:
                        # we only have 'query_toks' in example for training/dev sets

                        # fix for examples: we want to use the 'query_toks_no_value' field of the example which anonymizes
                        # values. However, it also anonymizes numbers (e.g. LIMIT 3 -> LIMIT 'value', which is not good
                        # since the official evaluator does expect a number and not a value
                        ex = fix_number_value(ex)

                        # we want the query tokens to be non-ambiguous (i.e. know for each column the table it belongs to,
                        # and for each table alias its explicit name)
                        # we thus remove all aliases and make changes such as:
                        # 'name' -> 'singer@name',
                        # 'singer AS T1' -> 'singer',
                        # 'T1.name' -> 'singer@name'
                        try:
                            query_tokens = disambiguate_items(ex['db_id'], ex['query_toks_no_value'],
                                                              self._tables_file, allow_aliases=False)
                        except Exception as e:
                            # there are two examples in the train set that are wrongly formatted, skip them
                            print(f"error with {ex['query']}")
                            print(e)

                    ins = self.text_to_instance(
                        utterance=ex['question'],
                        db_id=ex['db_id'],
                        sql=query_tokens)

                    if ins is not None:
                        yield ins
        else:
            if not file_path.endswith('.jsonl'):
                raise ConfigurationError(f"Don't know how to read filetype of {file_path}")

            logger.info("reading instance from file at: %s", file_path)
            with open(file_path, "r") as data_file:
                for _index, line in enumerate(data_file.readlines()):
                    line = line.strip("\n")
                    if not line:
                        continue
                    ex = json.loads(line)
                    query_tokens = None

                    if 'sql' in ex:
                        query_tokens = gen_wiki_tokens(ex, self._tables_file)
                    #print(query_tokens)
                    instance = self.text_to_instance(
                        utterance=ex['question'],
                        db_id=ex['table_id'],
                        sql=query_tokens)


                    if instance is not None:
                        yield instance

    @overrides
    def  text_to_instance(self,
                          utterance: str,
                          db_id: str,
                          sql: List[str] = None):
        fields: Dict[str,Field] = {}
        if self._is_spider:
            db_context = SpiderDBContext(db_id, utterance, tokenizer=self._tokenizer,
                                         tables_file=self._tables_file, dataset_path=self._dataset_path)
            table_field = SpiderKnowledgeGraphField(db_context.knowledge_graph,
                                                    db_context.tokenized_utterance,
                                                    self._utterance_token_indexers,
                                                    entity_tokens=db_context.entity_tokens,
                                                    include_in_vocab=False,  # TODO: self._use_table_for_vocab,
                                                    max_table_tokens=None)  # self._max_table_tokens)

            world = SpiderWorld(db_context,nl_context=None, query=sql)
            fields["utterance"] = TextField(db_context.tokenized_utterance, self._utterance_token_indexers)

            action_sequence, all_actions = world.get_action_sequence_and_all_actions()

            if action_sequence is None and self._keep_if_unparsable:
                # print("Parse error")
                action_sequence = []
            elif action_sequence is None:
                return None

            index_fields: List[Field] = []
            production_rule_fields: List[Field] = []

            for production_rule in all_actions:
                nonterminal, rhs = production_rule.split(' -> ')
                production_rule = ' '.join(production_rule.split(' '))
                field = ProductionRuleField(production_rule,
                                            world.is_global_rule(rhs),
                                            nonterminal=nonterminal)
                production_rule_fields.append(field)

            valid_actions_field = ListField(production_rule_fields)
            fields["valid_actions"] = valid_actions_field

            action_map = {action.rule: i  # type: ignore
                          for i, action in enumerate(valid_actions_field.field_list)}

            for production_rule in action_sequence:
                index_fields.append(IndexField(action_map[production_rule], valid_actions_field))
            if not action_sequence:
                index_fields = [IndexField(-1, valid_actions_field)]

            action_sequence_field = ListField(index_fields)
            fields["action_sequence"] = action_sequence_field
            fields["world"] = MetadataField(world)
            fields["schema"] = table_field
        else:
            db_context = WikiDBContext(db_id, utterance, tokenizer=self._tokenizer,
                                         tables_file=self._tables_file, dataset_path=self._dataset_path)
            print(db_context.entity_tokens)
            #todo 这个WikiKnowledgeGraphField和对应的spider的一模一样 只是改动了类名
            table_field= WikiKnowledgeGraphField(db_context.knowledge_graph,
                                                db_context.tokenized_utterance,
                                                self._utterance_token_indexers,
                                                entity_tokens=db_context.entity_tokens,
                                                include_in_vocab=False,  # TODO: self._use_table_for_vocab,
                                                max_table_tokens=None)  # self._max_table_tokens)

            world = WikiWorld(db_context,nl_context=None,query= sql)
            fields["utterance"] = TextField(db_context.tokenized_utterance, self._utterance_token_indexers)

            #todo 这一步会报错 应该是grammar不匹配的问题 ParseError:['select', '1-10015132-11@Position', 'where', '1-10015132-11@School/Club Team', '=', "'value'"]
            #todo action_sequence: None
            #todo all_actions: ['arg_list -> [expr, ",", arg_list]', 'arg_list -> [expr]', 'arg_list_or_star -> ["*"]', 'arg_list_or_star -> [arg_list]', 'binaryop -> ["!="]', 'binaryop -> ["*"]', 'binaryop -> ["+"]', 'binaryop -> ["-"]', 'binaryop -> ["/"]', 'binaryop -> ["<"]', 'binaryop -> ["<="]', 'binaryop -> ["<>"]', 'binaryop -> ["="]', 'binaryop -> [">"]', 'binaryop -> [">="]', 'binaryop -> ["and"]', 'binaryop -> ["like"]', 'binaryop -> ["or"]', 'boolean -> ["false"]', 'boolean -> ["true"]', 'col_ref -> ["1-10015132-11@col0"]', 'col_ref -> ["1-10015132-11@col1"]', 'col_ref -> ["1-10015132-11@col2"]', 'col_ref -> ["1-10015132-11@col3"]', 'col_ref -> ["1-10015132-11@col4"]', 'col_ref -> ["1-10015132-11@col5"]', 'column_name -> ["1-10015132-11@col0"]', 'column_name -> ["1-10015132-11@col1"]', 'column_name -> ["1-10015132-11@col2"]', 'column_name -> ["1-10015132-11@col3"]', 'column_name -> ["1-10015132-11@col4"]', 'column_name -> ["1-10015132-11@col5"]', 'expr -> [in_expr]', 'expr -> [source_subq]', 'expr -> [unaryop, expr]', 'expr -> [value, "between", value, "and", value]', 'expr -> [value, "like", string]', 'expr -> [value, binaryop, expr]', 'expr -> [value]', 'fname -> ["all"]', 'fname -> ["avg"]', 'fname -> ["count"]', 'fname -> ["max"]', 'fname -> ["min"]', 'fname -> ["sum"]', 'from_clause -> ["from", source]', 'from_clause -> ["from", table_name, join_clauses]', 'function -> [fname, "(", "distinct", arg_list_or_star, ")"]', 'function -> [fname, "(", arg_list_or_star, ")"]', 'group_clause -> [expr, ",", group_clause]', 'group_clause -> [expr]', 'groupby_clause -> ["group", "by", group_clause, "having", expr]', 'groupby_clause -> ["group", "by", group_clause]', 'in_expr -> [value, "in", expr]', 'in_expr -> [value, "in", string_set]', 'in_expr -> [value, "not", "in", expr]', 'in_expr -> [value, "not", "in", string_set]', 'iue -> ["except"]', 'iue -> ["intersect"]', 'iue -> ["union"]', 'join_clause -> ["join", table_name, "on", join_condition_clause]', 'join_clauses -> [join_clause, join_clauses]', 'join_clauses -> [join_clause]', 'join_condition -> [column_name, "=", column_name]', 'join_condition_clause -> [join_condition, "and", join_condition_clause]', 'join_condition_clause -> [join_condition]', 'limit -> ["limit", non_literal_number]', 'non_literal_number -> ["1"]', 'non_literal_number -> ["2"]', 'non_literal_number -> ["3"]', 'non_literal_number -> ["4"]', 'number -> ["value"]', 'order_clause -> [ordering_term, ",", order_clause]', 'order_clause -> [ordering_term]', 'orderby_clause -> ["order", "by", order_clause]', 'ordering -> ["asc"]', 'ordering -> ["desc"]', 'ordering_term -> [expr, ordering]', 'ordering_term -> [expr]', 'parenval -> ["(", expr, ")"]', 'query -> [select_core, groupby_clause, limit]', 'query -> [select_core, groupby_clause, orderby_clause, limit]', 'query -> [select_core, groupby_clause, orderby_clause]', 'query -> [select_core, groupby_clause]', 'query -> [select_core, orderby_clause, limit]', 'query -> [select_core, orderby_clause]', 'query -> [select_core]', 'select_core -> [select_with_distinct, select_results, from_clause, where_clause]', 'select_core -> [select_with_distinct, select_results, from_clause]', 'select_core -> [select_with_distinct, select_results, where_clause]', 'select_core -> [select_with_distinct, select_results]', 'select_result -> ["*"]', 'select_result -> [column_name]', 'select_result -> [expr]', 'select_result -> [table_name, ".*"]', 'select_results -> [select_result, ",", select_results]', 'select_results -> [select_result]', 'select_with_distinct -> ["select", "distinct"]', 'select_with_distinct -> ["select"]', 'single_source -> [source_subq]', 'single_source -> [table_name]', 'source -> [single_source, ",", source]', 'source -> [single_source]', 'source_subq -> ["(", query, ")"]', 'statement -> [query, iue, query]', 'statement -> [query]', 'string -> ["\'", "value", "\'"]', 'string_set -> ["(", string_set_vals, ")"]', 'string_set_vals -> [string, ",", string_set_vals]', 'string_set_vals -> [string]', "table_name -> ['1-10015132-11']", "table_source -> ['1-10015132-11']", 'unaryop -> ["+"]', 'unaryop -> ["-"]', 'unaryop -> ["not"]', 'value -> ["YEAR(CURDATE())"]', 'value -> [boolean]', 'value -> [column_name]', 'value -> [function]', 'value -> [number]', 'value -> [parenval]', 'value -> [string]', 'where_clause -> ["where", expr, where_conj]', 'where_clause -> ["where", expr]', 'where_conj -> ["and", expr, where_conj]', 'where_conj -> ["and", expr]']
            action_sequence, all_actions = world.get_action_sequence_and_all_actions()

            if action_sequence is None and self._keep_if_unparsable:
                # print("Parse error")
                action_sequence = []
            elif action_sequence is None:
                return None


            production_rule_fields: List[Field] = []

            for production_rule in all_actions:

                nonterminal, rhs = production_rule.split(' -> ')
                production_rule = ' '.join(production_rule.split(' '))

                field = ProductionRuleField(production_rule,
                                            world.is_global_rule(rhs),
                                            nonterminal=nonterminal)
                production_rule_fields.append(field)

            valid_actions_field = ListField(production_rule_fields)
            fields["valid_actions"] = valid_actions_field

            index_fields: List[Field] = []

            # action: ProductionRuleField
            action_map = {action.rule: i  # type: ignore
                          for i, action in enumerate(valid_actions_field.field_list)}

            for production_rule in action_sequence:
                index_fields.append(IndexField(action_map[production_rule], valid_actions_field))
            if not action_sequence:
                index_fields = [IndexField(-1, valid_actions_field)]

            action_sequence_field = ListField(index_fields)
            fields["action_sequence"] = action_sequence_field

            fields["world"] = MetadataField(world)
            fields["schema"] = table_field

            #print(fields)



        return Instance(fields)



'''
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
'''


