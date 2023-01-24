# Instructions - Predict Customer Churn with Clean Code

# Objective

This is a project to implement best coding practices. You will not need to code for this project entirely from scratch.

We have already provided you the ``churn_notebook.ipynb`` file containing the solution to identify credit card customers that are most likely to churn, but without implementing the engineering and software best practices.

You will need to refactor the given ``churn_notebook.ipynb`` file following the best coding practices to generate these files:

1. ``churn_library.py``
2. ``churn_script_logging_and_tests.py``
3. ``README.md``

## High-Level Instructions

These are the three files you will need to complete after refactoring the ``churn_notebook.ipynb`` file:

1. ``churn_library.py``

The ``churn_library.py`` is a library of functions to find customers who are likely to churn. You may be able to complete this project by completing each of these functions, but you also have the flexibility to change or add functions to meet the rubric criteria.

The document strings have already been created for all the functions in the ``churn_library.py`` to assist with one potential solution. In addition, for a better understanding of the function call, see the Sequence diagram.

After you have defined all functions in the ``churn_library.py``, you may choose to add an ``if __name__ == "__main__"`` block that allows you to run the code below and understand the results for each of the functions and refactored code associated with the original notebook.

``ipython churn_library.py``


2. ``churn_script_logging_and_tests.py``

This file should:

- Contain unit tests for the churn_library.py functions. You have to write test for each input function. Use the basic assert statements that test functions work properly. The goal of test functions is to checking the returned items aren't empty or folders where results should land have results after the function has been run.
- Log any errors and INFO messages. You should log the info messages and errors in a .log file, so it can be viewed post the run of the script. The log messages should easily be understood and traceable.

Also, ensure that testing and logging can be completed on the command line, meaning, running the below code in the terminal should test each of the functions and provide any errors to a file stored in the /logs folder.

``ipython churn_script_logging_and_tests.py``

**Testing framework**: As long as you fulfil all the rubric criteria, the choice of testing framework rests with the student. For instance, you can use [pytest](https://docs.pytest.org/en/7.1.x/getting-started.html#) for writing functional tests.

3. ``README.md``

This file will provide an overview of the project, the instructions to use the code, for example, it explains how to test and log the result of each function. For instance, you can have the following detailed sections in the ``README.md`` file:

- Project description
- Files and data description
- Running the files

## Code Quality Considerations

- **Style Guide** - Format your refactored code using [PEP 8 – Style Guide](https://peps.python.org/pep-0008/). Running the command below can assist with formatting. To assist with meeting pep 8 guidelines, use ``autopep8`` via the command line commands below:

```
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
autopep8 --in-place --aggressive --aggressive churn_library.py
```

- **Style Checking and Error Spotting** - Use [Pylint](https://pypi.org/project/pylint/) for the code analysis looking for programming errors, and scope for further refactoring. You should check the pylint score using the command below.

```
pylint churn_library.py
pylint churn_script_logging_and_tests.py
```

You should make sure you don't have any errors, and that your code is scoring as high as you can get it! Shoot for a pylint score exceeding 7 for both Python files.
