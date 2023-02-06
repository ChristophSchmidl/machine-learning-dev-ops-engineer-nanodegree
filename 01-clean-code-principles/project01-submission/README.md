# Predict Customer Churn

Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

Refactor the given **churn_notebook.ipynb** file following the best coding practices to generate these files:

- ``churn_library.py``
- ``churn_script_logging_and_tests.py``
- ``README.md``


## Project Description

This project contains the initial notebook called ``churn_notebook.ipynb`` that predicts churn of credit card customers based on the dataset ``data/bank_data.csv`` using logistic regression and random forest algorithm. The aim of this project is to refactor the given notebook into two files, namely:

- ``churn_library.py``: Contains the refactored code of the notebook as functions
- ``churn_script_logging_and_tests.py``: Contains tests based on the library ``pytest`` to test functions of ``churn_library.py``

To ensure correct formatting according to the PEP8 standard, the following libraries have been used:

- ``autopep8``
- ``pylint``

## Files and data description

The following sections explain the different folders of this project and their contents.

### data

The ``data``-folder contains the dataset called ``bank_data.csv`` that is used to train the different models. It contains features such as ``Customer_Age``, ``Marital_Status``, ``Income_Category`` besides other features.

### images

The ``images``-folder containts two sub-folders, namely ``eda`` and ``results``. The ``eda``-folder contains all images that are produced by the exploratory data analysis steps such as:

- ``churn_distribution.png``
- ``corr_heatmap.png``
- ``customer_age_distribution.png``
- ``marital_status_distribution.png``
- ``total_transaction_distribution.png``

The ``results``-folder contains all images that are produced by the training and evaluation steps of the two different models. The following images are produced:

- ``feature_importances.png``
- ``lr_results.png``
- ``rf_results.png``
- ``ROC_curves.png``

### logs

The ``logs``-folder contains a single log-file called ``churn_library.log`` that gets produced by running the test module ``churn_script_logging_and_tests.py`` and by running the file ``churn_library.py``.

### models

The ``models``-folder contains the saved models of the logistic regression and random forest algorithms, namely ``logistic_model.pkl`` and ``rfc_model.pkl``. Both models were saved using the ``joblib`` library and ``dump()``-method which is using the pickle protocol.


## Running Files

The easiest way to run the files is by creating a virtual environment first. Python version 3.9.13 has been used to create this project. You can execute the following steps to create a virtual environment, activate that environment and install all dependencies:

1. Create virtual environment with Python version 3.x.x: ``python3 -m venv venv``
2. Activate virtual environment (Linux): ``source venv/bin/activate``
3. With the activated environment, install all dependencies: ``pip install -r requirements``

### ``churn_library.py``

If you want to execute the whole pipeline in one go, you can just execute the file ``churn_library.py`` as usual with Python:

``(venv)> python churn_library.py``

### ``churn_script_logging_and_tests.py``

You can execute the test module ``churn_script_logging_and_tests.py`` using ``pytest``:

``(venv)> pytest churn_script_logging_and_tests.py -s``

Please keep in mind that executing the script directly with Python leads to a clean state were all images, logs and models are lost:

``(venv)> python churn_script_logging_and_tests.py``
