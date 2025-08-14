# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'molecular-simulations'
copyright = '2025, Matt Sinclair'
author = 'Matt Sinclair'
release = '2025'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown
# here.

# make sure sphinx always uses the current branch
import os
import sphinx_rtd_theme
import sys
import types

print('CONF LOADED; RTD =', os.environ.get('READTHEDOCS'))

sys.path.insert(0, os.path.abspath('../src'))

# add sphinx extensions and autodoc configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
]

autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
autoclass_content = 'both'
autodoc_typehints = 'description'

autodoc_mock_imports = [
    'numba', 'numba.njit', 'openbabel', 'openbabel.pybel',
    'parmed', 'pdbfixer', 'pdbfixer.PDBFixer', 'pdbfixer.pdbfixer',
    'pdbfixer.pdbfixer.Sequence'
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

templates_path = ['_templates']

# Configuration for intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "maplotlib": ("https://matplotlib.org/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "openmm": ("https://docs.openmm.org/latest/userguide/", None),
    "parsl": ("https://parsl.readthedocs.io/en/stable", None),
    "polars": ("https://docs.pola.rs/api/python/stable", None),
    "parmed": ("https://parmed.github.io/ParmEd/html/", None),
    "rdkit": ("https://rdkit.org/docs/", None),
    "dask": ("https://docs.dask.org/en/stable/", None),
}
