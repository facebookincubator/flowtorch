# Copyright (c) Meta Platforms, Inc
import re
import sys
from typing import Optional, List, Dict, Sequence

import flowtorch.docs.parse as parse

re_section_header = re.compile(r"\w+:")


"""
    NOTE: We don't support the following sections
        * "Arguments"
        * "Keyword Args"
        * "Keyword Arguments"
        * "Methods"
        * "Other Parameters"
        * "Parameters",
        * "Warns"
    since we are trying to simplify and make a MVP! I think these sections
    should be omitted from future versions too.

    TODO: In the future, give warning when alias of section is used or if
    inconsistent aliases are used throughout source.
"""
valid_headers = {
    "args": parse.args,
    "attributes": parse.attributes,
    "example": parse.example,
    "examples": parse.example,
    "note": parse.notes,
    "notes": parse.notes,
    "raise": parse.raises,
    "raises": parse.raises,
    "references": parse.references,
    "return": parse.returns,
    "returns": parse.returns,
    "see also": parse.see_also,
    "todo": parse.todo,
    "warning": parse.warning,
    "warnings": parse.warning,
    "yield": parse.yields,
    "yields": parse.yields,
}


def trim(docstring: str) -> str:
    """
    Trims whitespace from docstrings according to PEP257.
    """
    if not docstring:
        return ""
    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = docstring.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    indent = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < sys.maxsize:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)
    # Return a single string:
    return "\n".join(trimmed)


def join_blocks(blocks: Sequence[str]) -> str:
    """
    Removes identation from a list of strings and joins into a single string.
    """
    if not len(blocks):
        return ""

    first_line = blocks[0].split("\n")[0]
    indentation = parse.detect_indentation(first_line)
    new_blocks = []

    for block in blocks:
        lines = block.split("\n")
        new_lines = []
        for line in lines:
            if line.startswith(indentation):
                new_lines.append(line[len(indentation) :])
            else:
                # DEBUG
                print(blocks)

                raise ValueError(f"Invalid indentation on line:\n{line}")
        new_blocks.append("\n".join(new_lines))

    return "\n\n".join(new_blocks)


class Docstring(object):
    """
    Represents a Docstring in Google style and converts to MDX.
    """

    def __init__(self, docstring: str, throw_warnings: bool = False) -> None:
        # Save raw docstring
        self._docstring = trim(docstring)
        self._throw_warnings = throw_warnings

        # Keep track of warnings and errors
        # self._warnings = []
        # self._errors = []

        # Parse into sections
        self._markdown: Optional[str] = None
        self._sections: Dict[str, str] = {}
        self._section()

    def _section(self) -> None:
        """
        Separates the Google docstring for this object into sections.

        This method effectively operates as a finite-state machine.
        """
        if self._sections != {}:
            return

        # Break docstring into blocks separated by empty line
        blocks = self._docstring.split("\n\n")
        if not len(blocks):
            return

        """
        for idx, b in enumerate(blocks):
            print(f'block {idx}')
            print(b)
            print('')
        """

        # Extract the short description
        self._sections["short_description"] = blocks[0]

        # TODO: Validate the short description
        # Must be a single line of <= 80 chars ending in a period

        # Extract the longer description, if present
        description_blocks = []
        idx = 1
        for idx in range(1, len(blocks)):
            block_lines = blocks[idx].split("\n")
            if not re_section_header.fullmatch(block_lines[0]):
                description_blocks.append(blocks[idx])
            else:
                break

        # TODO: Validate the longer description including consistency
        # of indentation
        if len(description_blocks):
            self._sections["description"] = "\n\n".join(description_blocks)

        # Break up remaining docstrings into sections
        section = ""
        section_blocks: List[str] = []
        for jdx in range(idx, len(blocks)):
            block_lines = blocks[jdx].split("\n")
            if re_section_header.fullmatch(block_lines[0]):
                # Flush previous section
                if len(section_blocks) and len(section):
                    # Check that section blocks have consistent indentation and remove
                    content = join_blocks(section_blocks)

                    # TODO: Throw error when unknown section type
                    # TODO: Throw warning when non-preferred alias is used
                    parse_fn = valid_headers[section]
                    normalized_section, markdown = parse_fn(content)  # type: ignore

                    # TODO: Throw error when duplicates of sections
                    self._sections[normalized_section] = markdown

                # Start new section
                section = block_lines[0][:-1].lower()
                section_blocks = ["\n".join(block_lines[1:])]

            else:
                section_blocks.append(blocks[jdx])

        # Flush previous section
        if len(section_blocks) and len(section):
            # Check that section blocks have consistent indentation and remove
            content = join_blocks(section_blocks)

            # TODO: Throw error when unknown section type
            # TODO: Throw warning when non-preferred alias is used
            parse_fn = valid_headers[section]
            normalized_section, markdown = parse_fn(content)  # type: ignore

            # TODO: Throw error when duplicates of sections
            self._sections[normalized_section] = markdown

        # DEBUG
        # for k, v in self._sections.items():
        #    print("section:", k)
        #    print(v)
        #    print('')

    def __repr__(self) -> str:
        """
        Output docstring in MDX format
        """
        return self.to_mdx()

    def to_mdx(self) -> str:
        """
        (Lazily) converts parsed docstring to MDX
        """
        if self._markdown is None:
            paragraphs = []

            if "description" in self._sections:
                paragraphs.append(self._sections["description"])

            # TODO: Take out section in these since they'll be in a box
            if "notes" in self._sections:
                paragraphs.append(f"### Notes\n{self._sections['notes']}")
            if "warning" in self._sections:
                paragraphs.append(f"### Warning\n{self._sections['warning']}")

            for section in [
                "args",
                "attributes",
                "returns",
                "yields",
                "raises",
                "example",
                "see_also",
                "todo",
                "references",
            ]:
                if section in self._sections:
                    section_text = section.capitalize().replace("_", " ")
                    paragraphs.append(f"### {section_text}\n{self._sections[section]}")

            self._markdown = "\n\n".join(paragraphs)

        return self._markdown
