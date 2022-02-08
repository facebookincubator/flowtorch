# Copyright (c) Meta Platforms, Inc

import importlib
import inspect
import pkgutil
from collections import OrderedDict
from inspect import isclass, isfunction, signature
from types import ModuleType
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple


def get_decorators(function: Callable) -> Sequence[str]:
    """Returns list of decorators names

    Args:
        function (Callable): decorated method/function

    Return:
        List of decorators as strings

    Example:
        Given:

        @my_decorator
        @another_decorator
        def decorated_function():
            pass

        >>> get_decorators(decorated_function)
        ['@my_decorator', '@another_decorator']

    """
    source = inspect.getsource(function)
    index = source.find("def ")
    return [
        line.strip().split()[0]
        for line in source[:index].strip().splitlines()
        if line.strip()[0] == "@"
    ]


def generate_class_markdown(symbol_name: str, entity: Any) -> str:
    markdown = []

    # Parents (for classes, this is like signature)
    parents = []
    for b in entity.__bases__:
        parents.append(b.__module__ + "." + b.__name__)
    parents_str = ", ".join(parents)

    # Docstring
    # TODO: Parse docstring and extract short summary
    docstring = entity.__doc__ if entity.__doc__ is not None else "empty docstring"
    docstring = "\n".join(line.strip() for line in docstring.splitlines())

    # short_summary = "```short summary```\n"
    safe_name = symbol_name.replace("_", r"\_")

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
    # markdown.append(short_summary)
    markdown.append("</div>\n</div>\n\n</PythonClass>\n")
    markdown.append(f"```\n{docstring}\n```\n")

    # Methods for class
    members = inspect.getmembers(entity, predicate=inspect.isroutine)
    members = [
        (n, obj) for n, obj in members if n == "__init__" or not n.startswith("_")
    ]
    members = [(n, obj) for n, obj in members if type(obj) not in ["method_descriptor"]]

    # member_strs = []
    for member_name, member_object in members:
        # Try to unwrap class method and fetch decorators
        # decorators = []
        try:
            if hasattr(member_object, "__wrapped__"):
                # decorators = get_decorators(member_object)
                member_object = member_object.__wrapped__
        except Exception:
            pass

        # TODO: Prepend decorators to method name
        # for d in decorators:
        #    member_strs.append(d)

        markdown.append("<PythonMethod>\n")
        markdown.append(
            """<div className="doc-method-row">
<div className="doc-method-label"><span className="doc-symbol-label">member</span></div>
<div className="doc-method-signature">\n"""
        )

        # Some built-ins don't have a signature and throw exception...
        try:
            member_signature = str(signature(member_object))
        except ValueError:
            member_signature = "()"

        safe_member_signature = member_signature.replace("<", "&#60;").replace(
            ">", "&#62;"
        )
        # safe_member_signature = safe_member_signature.replace("'", "\'")

        safe_member_name = member_name.replace("_", r"\_")
        safe_member_id = member_name.replace("_", "-")
        markdown.append(
            f"""###  <span className="doc-symbol-name">{safe_member_name}</span>\
 {{#{safe_member_id}}}\n"""
        )
        markdown.append(
            f"""<span className="doc-symbol-signature">{safe_member_signature}\
</span>\n"""
        )
        markdown.append("</div>\n</div>\n\n</PythonMethod>\n")

        member_docstring = (
            member_object.__doc__
            if member_object.__doc__ is not None
            else "<empty docstring>"
        )
        member_docstring = "\n".join(
            line.strip() for line in member_docstring.splitlines()
        )
        markdown.append(f"```\n{member_docstring}\n```\n")

    return "\n".join(markdown)


def generate_module_markdown(symbol_name: str, entity: Any) -> str:
    markdown = []

    # Docstring
    # TODO: Parse docstring and extract short summary
    docstring = entity.__doc__ if entity.__doc__ is not None else "empty docstring"
    docstring = "\n".join(line.strip() for line in docstring.splitlines())

    # short_summary = "```short summary```\n"
    safe_name = symbol_name.replace("_", r"\_")

    # Create top section for class
    markdown.append("<PythonModule>\n")
    markdown.append(
        """<div className="doc-module-row">
<div className="doc-module-label"><span className="doc-symbol-label">module</span></div>
<div className="doc-module-signature">\n"""
    )

    markdown.append(
        f"""## <span className="doc-symbol-name">{safe_name}</span> {{#module}}\n"""
    )
    # markdown.append(short_summary)
    markdown.append("</div>\n</div>\n\n</PythonModule>\n")
    markdown.append(f"```\n{docstring}\n```\n")

    return "\n".join(markdown)


def generate_function_markdown(symbol_name: str, entity: Any) -> str:
    markdown = []

    # Docstring
    # TODO: Parse docstring and extract short summary
    docstring = entity.__doc__ if entity.__doc__ is not None else "empty docstring"
    docstring = "\n".join(line.strip() for line in docstring.splitlines())
    # short_summary = "```short summary```\n"

    # DEBUG
    try:
        if hasattr(entity, "__wrapped__"):
            # decorators = get_decorators(entity)
            entity = entity.__wrapped__
    except Exception:
        pass

    # TODO: Prepend decorators to method name
    # for d in decorators:
    #    member_strs.append(d)

    # Some built-ins don't have a signature and throw exception...
    try:
        entity_signature = str(signature(entity))
    except ValueError:
        entity_signature = "()"

    safe_name = symbol_name.replace("_", r"\_")
    safe_signature = entity_signature.replace("<", "&lt;").replace(">", "&gt;")

    markdown.append("<PythonFunction>\n")
    markdown.append(
        """<div className="doc-function-row">
<div className="doc-function-label"><span className="doc-symbol-label">function\
</span></div>
<div className="doc-function-signature">\n"""
    )

    markdown.append(
        f"""## <span className="doc-symbol-name">{safe_name}</span> \
{{#function}}\n"""
    )
    markdown.append(
        f"""<span className="doc-symbol-signature">{safe_signature}</span>"""
    )
    # markdown.append(short_summary)
    markdown.append("</div>\n</div>\n\n</PythonFunction>\n")
    markdown.append(f"```\n{docstring}\n```\n")

    return "\n".join(markdown)


def documentable_symbols(module: ModuleType) -> Sequence[Tuple[str, Any]]:
    """
    Given a module object, returns a list of object name and values for documentable
    symbols (functions and classes defined in this module or a subclass)
    """
    return [
        (n, m)
        for n, m in inspect.getmembers(module, None)
        if isfunction(m) or (isclass(m) and m.__module__.startswith(module.__name__))
    ]


def walk_packages(
    modname: str, filter: Optional[Callable[[Any], bool]] = None
) -> Mapping[str, Tuple[ModuleType, Sequence[Tuple[str, Any]]]]:
    """
    Given a base module name, return a mapping from the name of all modules
    accessible under the base to a tuple of module and symbol objects.

    A symbol is represented by a tuple of the object name and value, and is
    either a function or a class accessible when the module is imported.

    """
    module = importlib.import_module(modname)
    modules = {modname: (module, documentable_symbols(module))}

    # NOTE: I use path of flowtorch rather than e.g. flowtorch.bijectors
    # to avoid circular imports
    path = module.__path__  # type: ignore

    # The followings line uncovered a bug that hasn't been fixed in mypy:
    # https://github.com/python/mypy/issues/1422
    for importer, this_modname, _ in pkgutil.walk_packages(
        path=path,  # type: ignore  # mypy issue #1422
        prefix=f"{module.__name__}.",
        onerror=lambda x: None,
    ):
        # Conditions required for mypy
        if importer is not None:
            if isinstance(importer, importlib.abc.MetaPathFinder):
                finder = importer.find_module(this_modname, None)
            elif isinstance(importer, importlib.abc.PathEntryFinder):
                finder = importer.find_module(this_modname)
        else:
            finder = None

        if finder is not None:
            module = finder.load_module(this_modname)

        else:
            raise Exception("Finder is none")

        if module is not None:
            # Get all classes and functions imported/defined in module
            modules[this_modname] = (module, documentable_symbols(module))

            del module
            del finder

        else:
            raise Exception("Module is none")

    return modules


def sparse_module_hierarchy(mod_names: Sequence[str]) -> Mapping[str, Any]:
    # Make list of modules to search and their hierarchy, pruning entries that
    # aren't in mod_names
    results: Dict[str, Any] = OrderedDict()
    this_dict = results

    for module in sorted(mod_names):
        submodules = module.split(".")

        # Navigate to the correct insertion place for this module
        for idx in range(0, len(submodules)):
            submodule = ".".join(submodules[0 : (idx + 1)])
            if submodule in this_dict:
                this_dict = this_dict[submodule]

        # Insert module if it doesn't exist already
        this_dict.setdefault(module, {})

    return results
