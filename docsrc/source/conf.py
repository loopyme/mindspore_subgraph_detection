import os
import sys

sys.path.insert(0, os.path.abspath('../../src'))
sys.path.insert(0, os.path.abspath('../../src/SubgraphDetection'))
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.todo',
              'sphinx.ext.coverage',
              'sphinx.ext.mathjax',
              'sphinx.ext.ifconfig',
              'sphinx.ext.viewcode',
              'recommonmark',
              'sphinx_markdown_tables',
              'sphinx.ext.autosummary',
              "sphinx.ext.napoleon"]
autodoc_typehints = 'description'
templates_path = ['_templates']
source_suffix = ['.rst', '.md']
master_doc = 'index'
project = 'mindspore_subgraph_detection'
copyright = '2020, Peter'
author = 'Peter'
version = '0.0.1'
release = '0.0.1'
language = None
exclude_patterns = []
pygments_style = 'sphinx'
todo_include_todos = True
html_theme = "sphinx_rtd_theme"
html_theme_options = {"style_nav_header_background": "174C4F"}
html_static_path = ['_static']
html_sidebars = {
    '**': [
        'relations.html',
        'searchbox.html',
    ]
}
htmlhelp_basename = 'mindspore_subgraph_detectiondoc'
latex_elements = {
}
latex_documents = [
    (master_doc, 'mindspore_subgraph_detection.tex', 'mindspore\\_subgraph\\_detection Documentation',
     'Peter', 'manual'),
]
man_pages = [
    (master_doc, 'mindspore_subgraph_detection', 'mindspore_subgraph_detection Documentation',
     [author], 1)
]
texinfo_documents = [
    (master_doc, 'mindspore_subgraph_detection', 'mindspore_subgraph_detection Documentation',
     author, 'mindspore_subgraph_detection', 'One line description of project.',
     'Miscellaneous'),
]
