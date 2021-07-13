# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT


from flowtorch.docs import sorted_entity_names
from flowtorch.utils import InterfaceError


class TestInterface:
    def test_documentable_case_insensitivity(self):
        """
        Checks whether there are any two entities that are indistinguishable by
        case. E.g. "flowtorch.params" the module and "flowtorch.Params" the
        class. For producing the API docs on Windows and Mac OS systems, it is
        advisable to have entity names that are unique regardless of case.

        """
        equivalence_classes = {}
        for n in sorted_entity_names:
            equivalence_classes.setdefault(n.lower(), []).append(n)
        erroneous_equivalences = [
            f'{{{", ".join(v)}}}' for v in equivalence_classes.values() if len(v) > 1
        ]

        if len(erroneous_equivalences):
            error_string = "\t\n".join(erroneous_equivalences)
            raise InterfaceError(
                f"""Documentable entities must be unique irrespective of case. The \
following equivalences were found:
    {error_string}"""
            )
