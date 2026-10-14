import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "US House 2026 Election Forecast"
copyright = "2026, Yevhen Bondariev"
author = "Yevhen Bondariev"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]
