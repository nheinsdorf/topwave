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


import sphinx_material

sys.path.insert(0, os.path.abspath(".."))
import topwave





# -- Project information -----------------------------------------------------

project = 'topwave'
copyright = '2023, Niclas Heinsdorf'
author = 'Niclas Heinsdorf'

# The full version, including alpha/beta/rc tags
release = '0.1.04a'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    # "numpydoc",
]

autodoc_type_aliases = {'ComplexVector': 'ComplexVector',
                        'Complex2x2': 'Complex2x2',
                        'Coupling': 'Coupling',
                        'IntVector': 'IntVector',
                        'Matrix': 'Matrix',
                        'PeriodicSite': 'PeriodicSite', # this is because for dataclasses it's weird with typehints
                        'Real3x3': 'Real3x3',
                        'RealList': 'RealList',
                        'SquareMatrix': 'SquareMatrix',
                        'Vector': 'Vector',
                        'VectorList': 'VectorList'}

autosummary_generate = True
autoclass_content = "class"


python_apigen_default_groups = [
    ("class:.*", "Classes"),
    ("data:.*", "Variables"),
    ("function:.*", "Functions"),
    ("method:.*", "Methods"),
    ("classmethod:.*", "Class methods"),
    ("property:.*", "Properties"),
    (r"method:.*\.[A-Z][A-Za-z,_]*", "Constructors"),
    (r"method:.*\.__[A-Za-z,_]*__", "Special methods"),
    (r"method:.*\.__(init|new)__", "Constructors"),
    (r"method:.*\.__(str|repr)__", "String representation"),
    # (r"method:.*\.is_[a-z,_]*", "Tests"),
    # (r"property:.*\.is_[a-z,_]*", "Tests"),
]
python_apigen_default_order = [
    ("class:.*", 10),
    ("data:.*", 11),
    ("function:.*", 12),
    ("method:.*", 24),
    ("classmethod:.*", 22),
    ("property:.*", 30),
    (r"method:.*\.[A-Z][A-Za-z,_]*", 20),
    (r"method:.*\.__[A-Za-z,_]*__", 23),
    (r"method:.*\.__(init|new)__", 20),
    (r"method:.*\.__(str|repr)__", 21),
    # (r"method:.*\.is_[a-z,_]*", 31),
    # (r"property:.*\.is_[a-z,_]*", 32),
]

python_apigen_order_tiebreaker = "alphabetical"
python_apigen_case_insensitive_filesystem = False
python_apigen_show_base_classes = True

python_transform_type_annotations_pep585 = False

object_description_options = [
    ("py:.*", dict(include_rubrics_in_toc=True)),
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']




# -- HTML theme settings ------------------------------------------------

html_show_sourcelink = False
html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
}

extensions.append("sphinx_material")
html_theme_path = sphinx_material.html_theme_path()
html_context = sphinx_material.get_html_context()
html_theme = "sphinx_material"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['custom.css']

html_title = "topwave"
html_short_title = "topwave"
html_favicon = "_static/favicon-32x32.png"
html_logo = "_static/topwave-logo.png"

# Define a custom inline Python syntax highlighting literal
rst_prolog = """
.. role:: python(code)
   :language: python
   :class: highlight
"""


# Sphinx Immaterial theme options
html_theme_options = {
    "repo_url": "https://github.com/nheinsdorf/topwave",
    "repo_name": "nheinsdorf/topwave",
    "repo_type": "github",
    "globaltoc_collapse": True,
    "globaltoc_depth": 3,
    "version_dropdown": True,
    "color_primary": "pink",
    "color_accent": "red",
    "master_doc": False,
}

ipython_execlines = [
    "import math",
    "import numpy as np",
    "import matplotlib.pyplot as plt",
    "import topwave as tp",
    "from pymatgen.core.structure import Lattice, Structure"
]