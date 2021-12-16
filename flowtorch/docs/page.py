# Copyright (c) Meta Platforms, Inc
from typing import Mapping, Sequence

from flowtorch.docs.markdown import generate_class_markdown
from flowtorch.docs.symbol import Symbol


class Page:
    """
    Represents a page of MDX markdown for a module, class, or function
    """
    def __init__(self, symbol: Symbol, symbols: Mapping[str, Symbol], hierarchy: Mapping[str, Sequence[str]]):
        if symbol._type.name == "CLASS":
            self._mdx = generate_class_markdown(symbol, symbols, hierarchy)

        elif symbol._type.name == "MODULE":
            self._mdx = "(module)"

        elif symbol._type.name == "FUNCTION":
            self._mdx = "(function)"

        else:
            raise Exception("Invalid symbol type for Page object")

    def __repr__(self) -> str:
        return self._mdx
