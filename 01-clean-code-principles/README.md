# Clean Code Principles

## Lessons in this section:

- Introduction
- Coding Best Practices
- Working with Others Using Version Control
- Production Ready Code
- **Project: Predict Customer Churn with Clean Code**

### Introduction

**Key takeaways**

1. Covered the course prerequisites
2. Introduced clean code principles
3. Identifed stakeholders for clean code principles
4. Looked at a history of clean code principles
5. Prepared the tools and environment
6. Provided an overview of the course project

### Coding Best Practices 

**Key takeaways**

1. Writing clean and modular code
2. Refactoring code
    - Boolean variables: is_{} or has_{}
3. Optimizing code to be more efficient
    - Remove for-loops in favor of vectorization (NumPy)
4. Writing documentation
    - (Inline) Comments
    - Docstrings (for whole scripts with author and date but also for functions and classes)
    - Project Documentation (README.md)
5. Following PEP8 & Linting
    - Auto-PEP8 (``pip install autopep8``)
    - Linting (``pip install pylint``)

**Key terms**

- **Refactoring** - the process of writing code that improves its maintainability, speed, and readability without changing its functionality.
- **Modular** - the logical partition of software into smaller programs for the purpose of improved maintainability, speed, and readability.
- **Efficiency** - using the resources optimally where resources could be memory, CPU, time, files, connections, databases, etc. [Source]
- **Optimization** - a way of writing code to be more efficient.
- **Documentation** - written material or illustration that explains computer software.
- **Linting** - the automated checking of your source code for programmatic, syntactic, or stylistic errors. [Source]
- **PEP8** - a document providing guidelines and best practices for writing Python code.

### Working with Others Using Version Control

**Key takeaways**

1. Creating branches
2. Using git and Github for different workflows
3. Performing code reviews

**Key terms**

- ``git add`` - add any new or modified files to the index
- ``git commit -m`` - a new commit containing the current contents of the index and the given log message describing the changes
- ``git push`` - frequently used to move local code to the cloud version of the repository
- ``git checkout -b`` - create and move to a new branch
- ``git checkout`` - used to move across branches that have already been created
- ``git branch`` - lists all branches
- ``git status`` - lists the status of the files that are updated or new
- ``git pull`` - pull updates from Github (remote) to local
- ``git branch -d`` deletes local branch

### Production Ready Code

**Key takeaways**

1. Catching errors
2. Writing tests
3. Writing logs
4. Model drift
5. Automated vs. non-automated retraining

**Key terms**

- **Try-except blocks** - are used to check code for errors. Try will execute if no errors occur.
- **Testing** - checking that the outcome of your software matches the expected requirements
- **Logging** - tracking your production code for informational, warning, and error catching purposes
- **Model drift** - the change in inputs and outputs of a machine learning model over time
- **Automated retraining** - the automated process of updating production machine learning models
- **Non-automated retraining** - a human-centered process of updating production machine learning models