# Copyright (c) Meta Platforms, Inc

class Tutorial:
    def __init__(self, id, title, label, path):
        self._id = id
        self._title = title
        self._label = label
        self._path = path

        # TODO: Form GitHub and Colab URLs based on a constant somewhere in FlowTorch?
