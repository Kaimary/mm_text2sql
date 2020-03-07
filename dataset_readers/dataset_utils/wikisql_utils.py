"""
utility functions for reading wikisql dataset
"""

from typing import  Dict,List
from dataset_readers.dataset_utils.common import Table,TableColumn
from collections import  defaultdict


def read_wiki_dataset_schema(schema_path: str) -> Dict[str, Dict[str, Table]]:
    schemas: Dict[str, Dict[str, Table]] = defaultdict(dict)
    # todo  finish reading
    return {**schemas}
