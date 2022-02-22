# Copyright (c) Meta Platforms, Inc
from typing import Optional

class Tutorial:
    def __init__(self, id:str, title:str, label:Optional[str] = None, path:Optional[str] = None) -> None:
        # Set defaults for label and path
        if label is None:
            label = title
        if path is None:
            path = f"{id}.ipynb"

        self._id = id
        self._title = title
        self._label = label
        self._path = path

        # TODO: Form GitHub and Colab URLs based on a constant somewhere in FlowTorch?
        pass
