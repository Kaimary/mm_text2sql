# pylint: disable=anomalous-backslash-in-string
"""
A ``Text2SqlTableContext`` represents the SQL context in which an utterance appears
for the any of the text2sql datasets, with the grammar and the valid actions.
"""
from typing import List, Dict

from dataset_readers.dataset_utils.common import Table


GRAMMAR_DICTIONARY_WIKI = {}
GRAMMAR_DICTIONARY_WIKI["statement"] = ['(query ws iue ws query)', '(query ws)']
# GRAMMAR_DICTIONARY_WIKI["iue"] = ['"intersect"', '"except"', '"union"']
GRAMMAR_DICTIONARY_WIKI["query"] = ['(ws select_core)']


GRAMMAR_DICTIONARY_WIKI["select_core"] = ['(select_with_distinct ws select_results ws where_clause)',
                                     '(select_with_distinct ws select_results)']

GRAMMAR_DICTIONARY_WIKI["select_with_distinct"] = ['(ws "select" ws "distinct")', '(ws "select")']
GRAMMAR_DICTIONARY_WIKI["select_results"] = ['(ws select_result ws "," ws select_results)', '(ws select_result)']
GRAMMAR_DICTIONARY_WIKI["select_result"] = ['(table_source ws "@*")', 'expr', 'col_ref']

# GRAMMAR_DICTIONARY_WIKI["source_subq"] = ['("(" ws query ws ")")', '("(" ws query ws iue ws query ws ")")']
# GRAMMAR_DICTIONARY_WIKI["source_subq"] = ['("(" ws query ws ")" ws "as" ws name)', '("(" ws query ws ")")']
# GRAMMAR_DICTIONARY_WIKI["limit"] = ['("limit" ws non_literal_number)']

GRAMMAR_DICTIONARY_WIKI["where_clause"] = ['(ws "where" wsp expr ws where_conj)', '(ws "where" wsp expr)']
GRAMMAR_DICTIONARY_WIKI["where_conj"] = ['(ws "and" wsp expr ws where_conj)', '(ws "and" wsp expr)',
                                    '(ws "or" wsp expr ws where_conj)', '(ws "or" wsp expr)']
'''
GRAMMAR_DICTIONARY_WIKI["groupby_clause"] = ['(ws "group" ws "by" ws group_clause ws "having" ws expr)',
                                        '(ws "group" ws "by" ws group_clause)']
GRAMMAR_DICTIONARY_WIKI["group_clause"] = ['(ws expr ws "," ws group_clause)', '(ws expr)']
GRAMMAR_DICTIONARY_WIKI["orderby_clause"] = ['ws "order" ws "by" ws order_clause']
GRAMMAR_DICTIONARY_WIKI["order_clause"] = ['(ordering_term ws "," ws order_clause)', 'ordering_term']
GRAMMAR_DICTIONARY_WIKI["ordering_term"] = ['(ws expr ws ordering)', '(ws expr)']
GRAMMAR_DICTIONARY_WIKI["ordering"] = ['(ws "asc")', '(ws "desc")']
'''

GRAMMAR_DICTIONARY_WIKI["col_ref"] = ['(table_name ws "." ws column_name)', 'column_name']
GRAMMAR_DICTIONARY_WIKI["table_source"] = [
    # '(table_name ws "as" ws table_alias)',
    'table_name']
GRAMMAR_DICTIONARY_WIKI["table_name"] = ["table_alias"]
GRAMMAR_DICTIONARY_WIKI["table_alias"] = ['"t1"', '"t2"', '"t3"', '"t4"']
GRAMMAR_DICTIONARY_WIKI["column_name"] = []

GRAMMAR_DICTIONARY_WIKI["ws"] = ['~"\s*"i']
GRAMMAR_DICTIONARY_WIKI['wsp'] = ['~"\s+"i']

GRAMMAR_DICTIONARY_WIKI["expr"] = [
                              # 'in_expr',
                              # Binary expressions.
                              # '(value ws binaryop ws source_subq)',
                              '(value ws binaryop wsp expr)',
                              '(binaryop ws expr)',
                              # Unary expressions.
                              '(unaryop ws expr)',
                              # Not Like expressions
                              # '(value wsp "not like" ws source_subq)',
                              # '(value wsp "not like" wsp string)',
                              # Like expressions.
                              #  '(value wsp "like" ws source_subq)',
                              # '(value wsp "like" wsp string)',
                              # Between expressions.
                              '(value ws "between" ws source_subq)',
                              '(value ws "between" wsp value ws "and" wsp value)',
                              'source_subq',
                              'value' ]
GRAMMAR_DICTIONARY_WIKI["in_expr"] = ['(value wsp "not" wsp "in" wsp string_set)',
                                 '(value wsp "in" wsp string_set)',
                                 '(value wsp "not" wsp "in" wsp expr)',
                                 '(value wsp "in" wsp expr)']

GRAMMAR_DICTIONARY_WIKI["value"] = ['parenval', '"YEAR(CURDATE())"', 'number', 'boolean',
                               'function', 'col_ref', 'string']
GRAMMAR_DICTIONARY_WIKI["parenval"] = ['"(" ws expr ws ")"']
GRAMMAR_DICTIONARY_WIKI["function"] = ['(fname ws "(" ws "distinct" ws arg_list ws ")")',
                                  '(fname ws "(" ws arg_list_or_star ws ")")']

'''
GRAMMAR_DICTIONARY_WIKI["arg_list_or_star"] = ['arg_list', '"*"']
'''
GRAMMAR_DICTIONARY_WIKI["arg_list_or_star"] = ['arg_list', '(table_name ws "@*")']



GRAMMAR_DICTIONARY_WIKI["arg_list"] = ['(expr ws "," ws arg_list)', 'expr']
 # TODO(MARK): Massive hack, remove and modify the grammar accordingly
# GRAMMAR_DICTIONARY_WIKI["number"] = ['~"\d*\.?\d+"i', "'3'", "'4'"]
GRAMMAR_DICTIONARY_WIKI["non_literal_number"] = ['"10"', '"1"', '"2"', '"3"', '"4"', '"5"', '"8"']
GRAMMAR_DICTIONARY_WIKI["number"] = ['ws "value" ws']
GRAMMAR_DICTIONARY_WIKI["string_set"] = ['ws "(" ws string_set_vals ws ")"']
GRAMMAR_DICTIONARY_WIKI["string_set_vals"] = ['(string ws "," ws string_set_vals)', 'string']
# GRAMMAR_DICTIONARY_WIKI["string"] = ['~"\'.*?\'"i']
GRAMMAR_DICTIONARY_WIKI["string"] = ['"\'" ws "value" ws "\'"']
GRAMMAR_DICTIONARY_WIKI["fname"] = ['"count"', '"sum"', '"max"', '"min"', '"avg"', '"all"']
GRAMMAR_DICTIONARY_WIKI["boolean"] = ['"true"', '"false"']

# TODO(MARK): This is not tight enough. AND/OR are strictly boolean value operators.
GRAMMAR_DICTIONARY_WIKI["binaryop"] = ['"+"', '"-"', '"*"', '"/"', '"="', '"!="', '"<>"',
                                  '">="', '"<="', '">"', '"<"', '"and"', '"or"', '"like"']
GRAMMAR_DICTIONARY_WIKI["unaryop"] = ['"+"', '"-"', '"not"', '"not"']


def update_grammar_with_tables_wiki(grammar_dictionary: Dict[str, List[str]],
                               schema: Dict[str, Table]) -> None:
    table_names = sorted([f'"{table.lower()}"' for table in
                          list(schema.keys())], reverse=True)
    grammar_dictionary['table_name'] += table_names

    all_columns = set()
    for table in schema.values():
        all_columns.update([f'"{table.name.lower()}@{column.name.lower()}"' for column in table.columns if column.name != '*'])
    sorted_columns = sorted([column for column in all_columns], reverse=True)
    grammar_dictionary['column_name'] += sorted_columns


def update_grammar_to_be_table_names_free_wiki(grammar_dictionary: Dict[str, List[str]]):
    """
    Remove table names from column names, remove aliases
    """

    grammar_dictionary["column_name"] = []
    grammar_dictionary["table_name"] = []
    grammar_dictionary["col_ref"] = ['column_name']
    grammar_dictionary["table_source"] = ['table_name']

    del grammar_dictionary["table_alias"]

'''
def update_grammar_flip_joins(grammar_dictionary: Dict[str, List[str]]):
    """
    Remove table names from column names, remove aliases
    """

    # using a simple rule such as join_clauses-> [(join_clauses ws join_clause), join_clause]
    # resulted in a max recursion error, so for now just using a predefined max
    # number of joins
    grammar_dictionary["join_clauses"] = ['(join_clauses_1 ws join_clause)', 'join_clause']
    grammar_dictionary["join_clauses_1"] = ['(join_clauses_2 ws join_clause)', 'join_clause']
    grammar_dictionary["join_clauses_2"] = ['(join_clause ws join_clause)', 'join_clause']
'''