# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
#sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

import pathlib
import tomllib
with open(pathlib.Path(__file__).parent.parent.parent / "pyproject.toml", "rb") as f:
    toml = tomllib.load(f)
pyproject = toml["project"]

project = pyproject["name"]
copyright = f"2024, {pyproject['authors'][0]['name']}"
author = pyproject["authors"][0]["name"]

# The full version, including alpha/beta/rc tags
release = pyproject["version"]


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    #'sphinx.ext.duration',
    #'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
]
napoleon_google_docstring = False
autosummary_generate = True
autoclass_content = "both"
autodoc_inherit_docstrings = True
set_type_checking_flag = True
autosummary_imported_members = True

autodoc_mock_imports = ["numpy", "scipy", "matplotlib", "samplerate", "numexpr", "numba", "tensorly", "hypothesis", "pytest", "aspcol"]


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
