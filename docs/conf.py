project = "cytoGateR Python"
author = "Matthew Li, Harry Robertson, Nikhil Karthik, Jean Yang, Lijia Yu"
copyright = "2026, the cytoGateR authors"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

html_theme = "sphinx_book_theme"
html_title = "cytoGateR Python"
html_theme_options = {
    "repository_url": "https://github.com/SydneyBioX/cytoGateR-py",
    "use_repository_button": True,
    "show_toc_level": 2,
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]

exclude_patterns = ["_build", "**.ipynb_checkpoints"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# Execute the tutorial during every build so documentation examples cannot
# silently drift away from the package API.
nb_execution_mode = "force"
nb_execution_timeout = 600
nb_execution_raise_on_error = True
# The complete workflow uses the R package's bundled demo data, which lives in
# a sibling repository. Its saved, locally validated outputs are published but
# it is not re-executed on GitHub's standalone checkout.
nb_execution_excludepatterns = ["full_pipeline.ipynb"]

myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3
suppress_warnings = ["mystnb.unknown_mime_type"]
