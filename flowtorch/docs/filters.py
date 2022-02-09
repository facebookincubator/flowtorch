# Copyright (c) Meta Platforms, Inc
import re
from typing import Any, Mapping


def regexs(config: Any) -> Mapping[str, Any]:
    """
    Given a configuration dictionary, converts the include and exclude strings
    to compiled regular expressions.
    """

    # Default include/exclude rules
    patterns = {
        "include": {"modules": re.compile(r".+"), "symbols": re.compile(r".+")},
        "exclude": {"modules": re.compile(r""), "symbols": re.compile(r"")},
    }

    # Override rules based on configuration file
    if "filters" in config:
        filters = config["filters"]
        for clude, rules in filters.items():
            for rule, pattern in rules.items():
                if type(pattern) is list:
                    pattern = "|".join(pattern)
                patterns[clude][rule] = re.compile(pattern)

    # Define filters
    def ismodule(full_name: str) -> bool:
        return (
            patterns["include"]["modules"].fullmatch(full_name) is not None
            and patterns["exclude"]["modules"].fullmatch(full_name) is None
        )

    def issymbol(full_name: str) -> bool:
        return (
            patterns["include"]["symbols"].fullmatch(full_name) is not None
            and patterns["exclude"]["symbols"].fullmatch(full_name) is None
        )

    return {"module": ismodule, "symbol": issymbol}
