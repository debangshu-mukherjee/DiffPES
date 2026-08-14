"""Configure the static Sphinx documentation build.

Extended Summary
----------------
The configuration loads package metadata, enables MyST Markdown, and connects
the local test-reference extension. The documentation workflow regenerates
tutorial exports before Sphinx reads this tree.

Routine Listings
----------------
:func:`setup`
    Register the local autodoc callbacks.
"""

from __future__ import annotations

import os
import sys
import tomllib
from datetime import datetime
from pathlib import Path

from beartype.typing import Any, Dict, List, Optional, Tuple
from sphinx.application import Sphinx

# Keep the documentation build on the CPU.
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["BUILDING_DOCS"] = "1"


def _identity(function: Any) -> Any:
    """PRIVATE: Return one callable without modification.

    Parameters
    ----------
    function : Any
        Callable or descriptor supplied to the documentation decorator.

    Returns
    -------
    unchanged : Any
        The original object.

    Notes
    -----
    The documentation decorator uses this helper in its factory form.
    """
    unchanged: Any = function
    return unchanged


def _noop_decorator(*args: Any, **kwargs: Any) -> Any:
    """PRIVATE: Preserve a callable while Sphinx imports the package.

    Parameters
    ----------
    *args : Any
        Positional decorator arguments or one directly decorated callable.
    **kwargs : Any
        Keyword arguments supplied to the decorator factory.

    Returns
    -------
    decorator_or_callable : Any
        The input callable or an identity decorator.

    Notes
    -----
    The replacement keeps original docstrings available to autodoc. The
    package import occurs only after this replacement becomes active.
    """
    if len(args) == 1 and callable(args[0]) and not kwargs:
        decorator_or_callable: Any = args[0]
    else:
        decorator_or_callable = _identity
    return decorator_or_callable


import jaxtyping  # noqa: E402

jaxtyping.jaxtyped = _noop_decorator


# Add project paths
project_root = os.path.abspath("../..")
src_path = os.path.join(project_root, "src")
venv_bin_path = os.path.join(project_root, ".venv", "bin")

sys.path.insert(0, src_path)
sys.path.insert(0, project_root)
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ext")
)
if os.path.isdir(venv_bin_path):
    os.environ["PATH"] = os.pathsep.join(
        [venv_bin_path, os.environ.get("PATH", "")]
    )

# Read project metadata from pyproject.toml
pyproject_path = os.path.join(project_root, "pyproject.toml")
with open(pyproject_path, "rb") as f:
    pyproject_data = tomllib.load(f)

project = pyproject_data["project"]["name"]

authors_data = pyproject_data["project"]["authors"]
author = (
    authors_data[0]["name"]
    if isinstance(authors_data[0], dict)
    else authors_data[0]
)

project_copyright = f"{datetime.now().year}, {author}"
release = pyproject_data["project"]["version"]

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "test_links",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# MyST-Parser configuration for LaTeX math rendering
myst_enable_extensions = [
    "dollarmath",  # Enable $...$ and $$...$$ math delimiters
    "amsmath",  # Enable LaTeX math environments like \begin{equation}
    "colon_fence",  # Enable ::: and ```{directive} fenced directives
]
# "autodoc": silence the non-actionable "error while formatting signature"
# warnings emitted when sphinx_autodoc_typehints introspects generated Chex or
# Hypothesis test wrappers. These wrappers have no stable local signature for
# Sphinx to publish, while their original test method remains documented.
# "forward_reference": upstream jax annotations (e.g. AxisName on the P
# PartitionSpec re-export) carry forward references Sphinx cannot resolve.
suppress_warnings = [
    "myst.mathjax",
    "autodoc",
    "sphinx_autodoc_typehints.local_function",
    "sphinx_autodoc_typehints.forward_reference",
]

# Ensure MyST parses all dollar-delimited math correctly
myst_dmath_double_inline = True
myst_heading_anchors = 3

# MathJax configuration to ensure math renders properly
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "processEscapes": True,
        "processEnvironments": True,
    },
    "options": {
        "ignoreHtmlClass": "tex2jax_ignore",
        "processHtmlClass": (
            "tex2jax_process|mathjax_process|math|output_area|content"
        ),
    },
    "chtml": {
        "scale": 1.0,
        "minScale": 0.5,
    },
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
html_extra_path = ["_extra"] if os.path.isdir("_extra") else []

html_css_files = [
    "custom.css",
]

html_js_files = [
    "custom.js",
]

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#0066cc",
        "color-brand-content": "#0066cc",
    },
    "dark_css_variables": {
        "color-brand-primary": "#3399ff",
        "color-brand-content": "#3399ff",
    },
    "sidebar_hide_name": False,
    "source_repository": "https://github.com/debangshu-mukherjee/diffpes/",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings for NumPy style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True

# All runtime dependencies are installed during the docs build (RTD installs
# the [docs, test] extras), so no imports need to be mocked.
autodoc_mock_imports = []

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": False,
    "ignore-module-all": False,
}

autodoc_typehints = "signature"
autodoc_typehints_format = "short"
autodoc_typehints_description_target = "documented"
python_use_unqualified_type_names = True
typehints_fully_qualified = False
always_document_param_types = False
typehints_document_rtype = True
typehints_use_signature = True
typehints_use_signature_return = True
autodoc_preserve_defaults = True
autodoc_inherit_docstrings = True

# Nitpicky cross-reference checking (-n) is intentionally OFF: it flags many
# refs (jaxtyping/beartype type names autodoc cannot resolve) for little gain.
# The nitpick_ignore list below is staged for if/when strict -n is enabled; it
# is inert while nitpicky = False.
nitpicky = False

# Type aliases for jaxtyping annotations
napoleon_type_aliases = {
    'Float[Array, " "]': "scalar float",
    'Float[Array, " 3"]': "3D float array",
    'Float[Array, " 3 3"]': "3x3 float array",
    'Float[Array, " M 3"]': "Mx3 float array",
    'Float[Array, " N 3"]': "Nx3 float array",
    'Float[Array, " N 4"]': "Nx4 float array",
    'Int[Array, " "]': "scalar int",
    'Int[Array, " N"]': "N int array",
    'Int[Array, " 2"]': "2-element int array",
    'Num[Array, " N"]': "N numeric array",
    'Bool[Array, " "]': "scalar bool",
    'Complex[Array, " "]': "scalar complex",
    "scalar_float": "float",
    "scalar_int": "int",
    "scalar_num": "numeric",
}

typehints_defaults = "comma"
typehints_type_aliases = napoleon_type_aliases

# Keep these names ready for a future strict cross-reference build.
# jaxtyping/beartype type names autodoc cannot resolve as Python classes.
nitpick_ignore = [
    ("py:class", "Float"),
    ("py:class", "Array"),
    ("py:class", "Int"),
    ("py:class", "Num"),
    ("py:class", "Bool"),
    ("py:class", "Complex"),
    ("py:class", "jaxtyping.Float"),
    ("py:class", "jaxtyping.Array"),
    ("py:class", "beartype"),
    ("py:class", "jaxtyped"),
    ("py:obj", "beartype"),
    ("py:obj", "jaxtyped"),
]

# Intersphinx mapping for cross-references
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "equinox": ("https://docs.kidger.site/equinox/", None),
}

# -- Custom setup ------------------------------------------------------------


def _skip_member(
    _app: Sphinx,
    what: str,
    name: str,
    obj: object,
    skip: bool,
    _options: object,
) -> bool:
    """PRIVATE: Select members that autodoc must omit.

    Parameters
    ----------
    _app : Sphinx
        Active Sphinx application.
    what : str
        Category of the documented Python object.
    name : str
        Candidate member name.
    obj : object
        Candidate Python object.
    skip : bool
        Previous skip decision from autodoc.
    _options : object
        Active autodoc options.

    Returns
    -------
    skip_result : bool
        ``True`` when autodoc must omit the candidate member.

    Notes
    -----
    The callback hides imported annotation helpers and private members. It
    also hides tuple-field descriptors that duplicate class attributes.
    """
    skip_names: Tuple[str, ...] = (
        "Float",
        "Array",
        "Int",
        "Num",
        "Bool",
        "Complex",
        "beartype",
        "jaxtyped",
        "tree_flatten",
        "tree_unflatten",
    )
    if name in skip_names or (
        name.startswith("_") and not name.startswith("__")
    ):
        skip_result: bool = True
    elif what == "data" and hasattr(obj, "_field_types"):
        skip_result = True
    else:
        skip_result = skip
    return skip_result


def _process_signature(
    _app: Sphinx,
    _what: str,
    _name: str,
    _obj: object,
    _options: object,
    signature: Optional[str],
    return_annotation: Any,
) -> Tuple[Optional[str], Any]:
    """PRIVATE: Shorten jaxtyping arrays in one displayed signature.

    Parameters
    ----------
    _app : Sphinx
        Active Sphinx application.
    _what : str
        Category of the documented Python object.
    _name : str
        Complete name of the documented object.
    _obj : object
        Documented Python object.
    _options : object
        Active autodoc options.
    signature : Optional[str]
        Signature text produced by autodoc.
    return_annotation : Any
        Return annotation produced by autodoc.

    Returns
    -------
    processed : Optional[str]
        Signature with compact array names, or ``None``.
    returned_annotation : Any
        Unchanged return annotation.

    Notes
    -----
    The replacements affect displayed text only. They do not change runtime
    annotations or their validation.
    """
    processed: Optional[str] = signature
    if processed:
        processed = processed.replace('Float[Array, " ', "FloatArray[")
        processed = processed.replace('Int[Array, " ', "IntArray[")
        processed = processed.replace('Bool[Array, " ', "BoolArray[")
        processed = processed.replace('Num[Array, " ', "NumArray[")
        processed = processed.replace('Complex[Array, " ', "ComplexArray[")
        processed = processed.replace('"]', "]")
    result: Tuple[Optional[str], Any] = (processed, return_annotation)
    return result


def _rewrite_included_asset_paths(
    app: Sphinx,
    relative_path: Path,
    parent_docname: str,
    content: List[str],
) -> None:
    """PRIVATE: Repoint repository image paths for included Markdown files.

    Parameters
    ----------
    app : Sphinx
        Active Sphinx application.
    relative_path : Path
        Path of the included file, relative to the source directory.
    parent_docname : str
        Document that pulled the file in through an include directive.
    content : List[str]
        Single-element list holding the included file text; mutated in
        place.

    Notes
    -----
    README.md references its figures as ``docs/source/_static/readme/...``
    so they render on GitHub and in IDE previews. Raw ``<img>`` tags pass
    through MyST untouched, and that repository-relative prefix points
    nowhere in the built site, where the same files are served from
    ``_static/``. Rewriting the prefix at include time keeps one set of
    paths in the README while the landing page loads the copies Sphinx
    ships under ``_static/readme/``.
    """
    if relative_path.name == "README.md":
        content[0] = content[0].replace(
            'src="docs/source/_static/', 'src="_static/'
        )


def setup(app: Sphinx) -> Dict[str, object]:
    """Register the local autodoc callbacks.

    Parameters
    ----------
    app : Sphinx
        Active Sphinx application.

    Returns
    -------
    metadata : Dict[str, object]
        Configuration version and parallel-read safety metadata.

    Notes
    -----
    Sphinx calls this function after it evaluates the configuration module.
    """
    app.connect("autodoc-skip-member", _skip_member)
    app.connect("autodoc-process-signature", _process_signature)
    app.connect("include-read", _rewrite_included_asset_paths)
    metadata: Dict[str, object] = {
        "version": "1.0",
        "parallel_read_safe": True,
    }
    return metadata


__all__: list[str] = ["setup"]
