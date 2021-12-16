# Copyright (c) Meta Platforms, Inc
from typing import Mapping, Sequence

from flowtorch.docs.symbol import Symbol
from flowtorch.docs.docstring import Docstring

def generate_class_markdown(symbol: Symbol, symbols: Mapping[str, Symbol], hierarchy: Mapping[str, Sequence[str]]) -> str:
    assert symbol._type.name == "CLASS"

    markdown = []

    # Parents (for classes, this is like signature)
    parents = []
    for b in symbol._bases:
        parents.append(b.__module__ + "." + b.__name__)
    parents_str = ", ".join(parents)

    # Docstring
    parsed_docstring = Docstring(symbol._docstring)

    short_summary = parsed_docstring._sections["short_description"]
    name = symbol._name.split('.')[-1]
    safe_name = name.replace("_", r"\_")

    # Create top section for class
    markdown.append("<PythonClass>\n")
    markdown.append(
        """<div className="doc-class-row">
<div className="doc-class-label"><span className="doc-symbol-label">class</span></div>
<div className="doc-class-signature">\n"""
    )
    markdown.append(
        f"""## <span className="doc-symbol-name">{safe_name}</span> {{#class}}"""
    )
    markdown.append(
        f"""<span className="doc-inherits-from">Inherits from: <span className=\
"doc-symbol-name">{parents_str}</span></span>\n"""
    )
    markdown.append(short_summary)
    markdown.append("</div>\n</div>\n\n</PythonClass>\n")

    mdx_docstring = parsed_docstring.to_mdx()
    if len(mdx_docstring):
        markdown.append(f"```\n{mdx_docstring}\n```\n")

    # Methods for class
    if symbol._name in hierarchy:
        member_names = hierarchy[symbol._name]

        # TODO: Filter out methods that are inherited

        for member_name in member_names:
            member = symbols[member_name]        

            markdown.append("<PythonMethod>\n")
            markdown.append(
                """<div className="doc-method-row">
    <div className="doc-method-label"><span className="doc-symbol-label">member</span></div>
    <div className="doc-method-signature">\n"""
            )

            # Some built-ins don't have a signature and throw exception...
            try:
                member_signature = str(member._signature)
            except ValueError:
                member_signature = "()"

            safe_member_signature = member_signature.replace("<", "&#60;").replace(
                ">", "&#62;"
            )
            parsed_member_docstring = Docstring(member._docstring)
            short_summary = parsed_member_docstring._sections["short_description"]

            safe_member_name = member._name.replace("_", r"\_")
            safe_member_id = member._name.replace("_", "-")
            markdown.append(
                f"""###  <span className="doc-symbol-name">{safe_member_name}</span>\
    {{#{safe_member_id}}}\n"""
            )
            markdown.append(
                f"""<span className="doc-symbol-signature">{safe_member_signature}\
    </span>\n"""
            )
            markdown.append(short_summary)
            markdown.append("</div>\n</div>\n\n</PythonMethod>\n")

            mdx_member_docstring = parsed_member_docstring.to_mdx()
            if len(mdx_member_docstring):
                markdown.append(f"```\n{parsed_member_docstring.to_mdx()}\n```\n")

    return "\n".join(markdown)


# def generate_module_markdown(symbol_name: str, entity: Any) -> str:
#     markdown = []

#     # Docstring
#     # TODO: Parse docstring and extract short summary
#     docstring = entity.__doc__ if entity.__doc__ is not None else "empty docstring"
#     docstring = "\n".join(line.strip() for line in docstring.splitlines())

#     # short_summary = "```short summary```\n"
#     safe_name = symbol_name.replace("_", r"\_")

#     # Create top section for class
#     markdown.append("<PythonModule>\n")
#     markdown.append(
#         """<div className="doc-module-row">
# <div className="doc-module-label"><span className="doc-symbol-label">module</span></div>
# <div className="doc-module-signature">\n"""
#     )

#     markdown.append(
#         f"""## <span className="doc-symbol-name">{safe_name}</span> {{#module}}\n"""
#     )
#     # markdown.append(short_summary)
#     markdown.append("</div>\n</div>\n\n</PythonModule>\n")
#     markdown.append(f"```\n{docstring}\n```\n")

#     return "\n".join(markdown)


# def generate_function_markdown(symbol_name: str, entity: Any) -> str:
#     markdown = []

#     # Docstring
#     # TODO: Parse docstring and extract short summary
#     docstring = entity.__doc__ if entity.__doc__ is not None else "empty docstring"
#     docstring = "\n".join(line.strip() for line in docstring.splitlines())
#     # short_summary = "```short summary```\n"

#     # DEBUG
#     try:
#         if hasattr(entity, "__wrapped__"):
#             # decorators = get_decorators(entity)
#             entity = entity.__wrapped__
#     except Exception:
#         pass

#     # TODO: Prepend decorators to method name
#     # for d in decorators:
#     #    member_strs.append(d)

#     # Some built-ins don't have a signature and throw exception...
#     try:
#         entity_signature = str(signature(entity))
#     except ValueError:
#         entity_signature = "()"

#     safe_name = symbol_name.replace("_", r"\_")
#     safe_signature = entity_signature.replace("<", "&lt;").replace(">", "&gt;")

#     markdown.append("<PythonFunction>\n")
#     markdown.append(
#         """<div className="doc-function-row">
# <div className="doc-function-label"><span className="doc-symbol-label">function\
# </span></div>
# <div className="doc-function-signature">\n"""
#     )

#     markdown.append(
#         f"""## <span className="doc-symbol-name">{safe_name}</span> \
# {{#function}}\n"""
#     )
#     markdown.append(
#         f"""<span className="doc-symbol-signature">{safe_signature}</span>"""
#     )
#     # markdown.append(short_summary)
#     markdown.append("</div>\n</div>\n\n</PythonFunction>\n")
#     markdown.append(f"```\n{docstring}\n```\n")

#     return "\n".join(markdown)
