# Maintaining the documentation

The website is built from function docstrings, pages under `docs/`, and
notebooks under `examples/`.

## Document a function

Write a NumPy-style docstring directly below the function definition:

```python
def my_function(data, threshold=0.5):
    """Describe what the function does in one sentence.

    Parameters
    ----------
    data : pandas.DataFrame
        Explain the expected rows, columns, and important constraints.
    threshold : float, default=0.5
        Explain how the parameter changes the calculation.

    Returns
    -------
    pandas.Series
        Explain the returned values and their index.

    Raises
    ------
    ValueError
        Explain when this error is raised.
    """
```

If the function is newly public, export it from `src/cytogater/__init__.py` and
add an `autofunction` entry to `docs/api.md`:

```rst
.. autofunction:: cytogater.my_function
```

Sphinx reads the function signature and docstring each time the website is
built; there is no separate API text to keep in sync.

## Update the full pipeline tutorial

Activate the package environment, install the current checkout, and execute
the canonical notebook in place:

```bash
conda activate cytogater-python
python -m pip install -e ".[plot,notebook]"
jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1200 \
  --ExecutePreprocessor.allow_errors=False \
  examples/full_pipeline.ipynb
```

Review every cell and confirm there are no errors or unexpected warnings. The
file `docs/full_pipeline.ipynb` is a symbolic link to this notebook, so the
website automatically uses the newly saved tables and figures.

The website does not re-execute this notebook on GitHub because the source R
demo data lives in a sibling repository. It does execute the self-contained
`docs/tutorial.md` example on every build.

## Preview the website

Install the documentation tools and run a strict build:

```bash
python -m pip install -e ".[doc]"
sphinx-build -b html -W --keep-going docs docs/_build/html
python -m http.server --directory docs/_build/html 8000
```

Open <http://localhost:8000> and inspect the tutorial and API reference.

## Publish

Commit the source files and push them to `main`:

```bash
git add docs examples/full_pipeline.ipynb src/cytogater README.md pyproject.toml
git commit -m "Update API documentation and full pipeline tutorial"
git push origin main
```

The `docs` GitHub Actions workflow builds and deploys the website.
