Project
==============================

UCL GCRF Project

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── analysis       <- Scripts to turn raw data into features for modeling
    │   │   └── correlation.py
    │   │   └── feature_selection.py
    │   │   └── model_selection.py
    │   │   └── linear_regression.py
    │   │   └── multicollinearity.py
    │   │   └── outliers.py
    │   │   └── spatial_autocorrelation.py
    │   │   
    │   ├── config         <- Provides a way to quickly access features and data sets     
    │   │   ├── config.py.py
    │   │ 
    │   └── raw_data       <- Process raw CDR data from tsv's to timestamped files
    │   │
    │   └── visualisation  <- Visualise some of the distributions of raw data
    │       └── raw_analysis.py
    │       └── raw_cdr.py
    │   │
    │   └── metrics        <- Compute various CDR and DHS metrics of the raw data
    │       └── adj_matrix.py
    │       └── aggregation.py
    │       └── cdr_derived.py
    │       └── cdr_fundamentals.py
    │       └── dhs_derived.py
    │       └── dhs_fundamentals.py
    │       └── spatial_lag.py  
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
