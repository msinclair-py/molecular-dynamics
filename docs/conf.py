# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
project = 'molecular-simulations'
copyright = '2025, Matt Sinclair'
author = 'Matt Sinclair'
release = '0.3.28'
version = '0.3'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',      # Add source code links
    'sphinx.ext.todo',          # Support TODO notes
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',        # Copy button for code blocks (add to deps)
]

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = False

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',  # Keep source order
    'special-members': '__init__',
    'exclude-members': '__weakref__',
}
autoclass_content = 'both'  # Include __init__ docstring
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# Napoleon settings (for NumPy/Google style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True

# Mock imports for packages that may not be installed during doc build
autodoc_mock_imports = [
    'numba',
    'openbabel',
    'parmed',
    'pdbfixer',
    'openmm',
    'MDAnalysis',
    'mdtraj',
    'parsl',
    'polars',
    'rdkit',
    'rust_simulation_tools',
    'seaborn',
    'sklearn',
    'scipy',
    'numpy',
    'natsort',
]

# Templates
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

# Custom CSS (create _static/custom.css if needed)
# html_css_files = ['custom.css']

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'MDAnalysis': ('https://docs.mdanalysis.org/stable/', None),
    'parsl': ('https://parsl.readthedocs.io/en/stable/', None),
    'polars': ('https://docs.pola.rs/api/python/stable/', None),
}

# -- Extension configuration -------------------------------------------------

# sphinx_copybutton settings
copybutton_prompt_text = r'>>> |\.\.\. |\$ '
copybutton_prompt_is_regexp = True

# TODO extension
todo_include_todos = True

# -- Suppress specific warnings ----------------------------------------------
suppress_warnings = ['autodoc.import']
