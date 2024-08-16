import sceptr

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "sceptr"
copyright = "2024, Yuta Nagano"
author = "Yuta Nagano"
release = sceptr.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinx.ext.napoleon"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_logo = "../sceptr.svg"
html_static_path = ['_static']
html_css_files = [
    'css/colours.css',
]
html_theme_options = {
    "repository_url": "https://github.com/yutanagano/sceptr",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
}
html_static_path = ["_static"]
