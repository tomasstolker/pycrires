# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('../'))


# -- Project information -----------------------------------------------------

project = 'pycrires'
copyright = '2021-2023, Tomas Stolker'
author = 'Tomas Stolker'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_automodapi.automodapi',
    'nbsphinx'
]

numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['.ipynb_checkpoints/*']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'sphinx_book_theme'
#
# html_theme_options = {
#     'path_to_docs': 'docs',
#     'repository_url': 'https://github.com/tomasstolker/pycrires',
#     'repository_branch': 'main',
#     'use_edit_page_button': True,
#     'use_issues_button': True,
#     'use_repository_button': True,
#     'use_download_button': True,
# }

html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    'github_url': 'https://github.com/tomasstolker/pycrires',
    'use_edit_page_button': True,
}

html_context = {
    "github_user": "tomasstolker",
    "github_repo": "pycrires",
    "github_version": "main",
    "doc_path": "docs",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []
