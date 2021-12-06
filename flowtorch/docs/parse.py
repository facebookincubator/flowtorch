# Copyright (c) Meta Platforms, Inc
import re
from typing import Tuple

re_argument = re.compile(r"^((\w+( \(\w+\))?)|(\*args)|(\*\*kwargs)):")
re_identation = re.compile(r"^\s*")
re_return_type = re.compile(r"^\w+:")


def detect_indentation(line: str) -> str:
    """
    Detects the identation at the start a string.
    """
    match = re_identation.match(line)
    if match:
        return match.group(0)
    else:
        return ""


# TODO: Extra arg(s) to parse functions for additional info?


def args(content: str) -> Tuple[str, str]:
    # TODO: Pass entity/symbol to args and check that all/only defined args
    # are documented. Also, that args are in same order as function/method

    # Break down content into args, types, and descriptions
    lines = content.split("\n")
    args = []
    this_arg = None
    this_indentation = ""
    this_lines = []

    for line in lines:
        # If this line begins with an argument...
        match = re_argument.match(line)
        if match:
            # ...process any previous ones
            if this_arg is not None:
                # TODO: Break up this_arg into arg and type
                args.append((this_arg, " ".join(this_lines).strip()))

                # Reset stack
                this_lines = []
                this_arg = None

            # ...and start recording this one
            this_arg = match.group(0)
            this_lines = [line[len(this_arg) :].strip()]
            this_arg = this_arg[:-1].strip()

        # If this line doesn't begin with an argument
        else:
            # ...it's an error if we haven't started recording
            if this_arg is None:
                raise ValueError(f"Invalid argument: {line}")

            # ...otherwise, try and add this line to argument
            if len(this_lines) == 1:
                this_indentation = detect_indentation(line)

            # TODO: More informative error message
            assert (
                len(line) >= len(this_indentation)
                and line[0 : len(this_indentation)] == this_indentation
            )
            line = line[len(this_indentation) :].strip()

            this_lines.append(line)

    if this_arg is not None:
        # TODO: Break up this_arg into arg and type
        args.append((this_arg, " ".join(this_lines).strip()))

    # Convert into MDX
    markdown = []
    for arg_type, description in args:
        markdown_line = f"* `{arg_type}`: {description}"
        markdown.append(markdown_line)

    return "args", "\n".join(markdown).strip()


def attributes(content: str) -> Tuple[str, str]:
    # TODO: Use same parsing as for args!
    markdown = content
    return "attributes", markdown


def example(content: str) -> Tuple[str, str]:
    markdown = content
    return "example", markdown


def notes(content: str) -> Tuple[str, str]:
    # TODO: Put notes in a box
    markdown = content
    return "notes", markdown


def raises(content: str) -> Tuple[str, str]:
    # TODO: Factor out common code with args()
    # Break down content into exceptions and descriptions
    lines = content.split("\n")
    exceptions = []
    this_exception = None
    this_indentation = ""
    this_lines = []

    for line in lines:
        # If this line begins with an argument...
        match = re_return_type.match(line)
        if match:
            # ...process any previous ones
            if this_exception is not None:
                # TODO: Break up this_arg into arg and type
                exceptions.append((this_exception, " ".join(this_lines).strip()))

                # Reset stack
                this_lines = []
                this_exception = None

            # ...and start recording this one
            this_exception = match.group(0)
            this_lines = [line[len(this_exception) :].strip()]
            this_exception = this_exception[:-1].strip()

        # If this line doesn't begin with an argument
        else:
            # ...it's an error if we haven't started recording
            if this_exception is None:
                raise ValueError(f"Invalid argument: {line}")

            # ...otherwise, try and add this line to argument
            if len(this_lines) == 1:
                this_indentation = detect_indentation(line)

            # TODO: More informative error message
            assert (
                len(line) >= len(this_indentation)
                and line[0 : len(this_indentation)] == this_indentation
            )
            line = line[len(this_indentation) :].strip()

            this_lines.append(line)

    if this_exception is not None:
        exceptions.append((this_exception, " ".join(this_lines).strip()))

    # Convert into MDX
    markdown = []
    for exception, description in exceptions:
        markdown_line = f"* `{exception}`: {description}"
        markdown.append(markdown_line)

    return "raises", "\n".join(markdown).strip()


def references(content: str) -> Tuple[str, str]:
    # TODO: Convert BibTex labels into links to bibliography section of website
    markdown = content
    return "references", markdown


def returns(content: str, section_type: str = "returns") -> Tuple[str, str]:
    # TODO: Break down content into args, types, and descriptions
    lines = content.split("\n")

    this_indentation = None
    this_lines = []

    # Extract optional type from first line
    match = re_return_type.match(lines[0])
    if match:
        this_type = match.group(0)[:-1].strip()
        this_lines = [lines[0][len(match.group(0)) :].strip()]
    else:
        this_type = ""
        this_indentation = ""
        this_lines = [lines[0].strip()]

    for line in lines[1:]:
        # If this line begins with an argument then throw error
        match = re_argument.match(line)
        if match:
            raise ValueError(f"Invalid line in '{section_type}' section: {line}")

        if this_indentation is None:
            this_indentation = detect_indentation(line)

        assert (
            len(line) >= len(this_indentation)
            and line[0 : len(this_indentation)] == this_indentation
        )
        this_lines.append(line[len(this_indentation) :].strip())

    # Convert to MDX
    type_str = f"`{this_type}`: " if this_type is not None else ""
    description = " ".join(this_lines).strip()
    markdown = f"* {type_str}{description}"

    return section_type, markdown


def see_also(content: str) -> Tuple[str, str]:
    # TODO: What formatting should be applied here?
    markdown = content
    return "see_also", markdown


def todo(content: str) -> Tuple[str, str]:
    # TODO: Put in a box?
    markdown = content
    return "todo", markdown


def warning(content: str) -> Tuple[str, str]:
    # TODO: Put in a box?
    markdown = content
    return "warning", markdown


def yields(content: str) -> Tuple[str, str]:
    return returns(content, section_type="yields")
