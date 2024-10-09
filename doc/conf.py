# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'HERMES: Hydrogen Economy Routing Model for cost-efficient Supply'
copyright = '2024, Uwe Langenmayr'
author = 'Uwe Langenmayr'
release = '2024'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
html_theme_options = {
    'body_max_width': "none",
    'page_width': 'auto',
}

html_css_files = [
    'custom.css'
]

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "HERMES"

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = "HERMES"

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = "PATH"
