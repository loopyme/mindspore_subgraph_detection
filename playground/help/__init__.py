from typing import List, Any

from tabulate import tabulate


def print_list(my_list: List[Any]):
    _ = list(map(print, my_list))


def print_table(my_list: List[Any], **kwargs):
    print(tabulate(my_list, **kwargs))
