# ML Model Scoring and Monitoring

## Lessons in this section:

- Welcome to ML Model Scoring and Monitoring
- Model Training and Deployment
- Model Scoring and Model Drift
- Diagnosing and Fixing Operational Problems
- Model Reporting and Monitoring with API's
- **Project: A Dynamic Risk Assessment System**

### Welcome to ML Model Scoring and Monitoring

**Key takeaways**

1. **Prerequisites**: you should be familiar with Python and machine learning before beginning the course.
2. **History of ML Scoring and Monitoring**: machine learning has existed for centuries, but the last few decades have had the most exciting new developments.
3. **When to use ML Model Scoring and Monitoring**: you should use the skills in this course every time you have a deployed ML model that will be used regularly. You may not need to use these skills if your model is only needed for one use, or if you're only doing exploration.
4. **Stakeholders**: company leaders, data scientists, ML engineers, and data engineers all have a need to be involved with ML Model Scoring and Monitoring processes.
5. **What you'll build**: your final project will be a dynamic risk assessment system that you'll build from the ground up.

### Model Training and Deployment

**Key takeaways**

- automatically ingest data, for use in the model, and for model training
- keep records related to ML processes, including data ingestion
- automate ML processes using cron jobs
- retrain and re-deploy ML models

**Key terms**

- **os**: a module you can use to access workspace directories and files from a Python script
- **ingestion**: the process of finding, gathering, recording, cleaning, and providing data as input to an ML project.
- **timestamp**: a character string recording a particular date and time
- **datetime**: a module containing capabilities for recording timestamps
- **crontab**: the file on a Linux machine that contains cron jobs
- **cron job**: a one-line code snippet that schedules a particular task
- **pickle**: the module used to read and write trained ML models
- **logistic regression**: an ML method used for categorical (0-1) classifications
- **re-deployment**: the process of overwriting a deployed ML model with a newer, improved version
- **dump()**: the method in the pickle module used to save a trained ML model
- **merge()**: a method for combining two datasets - also including an option to record which entries are unique to particular datasets, and which are common across both
- **distributed file system**: a collection of machines that allow data to be spread across multiple locations, to make work with extremely large datasets more feasible
- **client/server model**: a hierarchical model allowing one machine to perform executive functions and control others, for more efficient data processing
- **MapReduce**: a framework for performing operations on distributed datasets

### Model Scoring and Model Drift

**Key takeaways**

1. **Automatic model scoring**: how to read data and score models automatically
2. **Recording model scores**: how to keep persistent records of model scores in your workspace
3. **Model drift**: how to perform several different tests to check for model drift
4. **Hypothesis testing**: how to use statistical tests to compare two different models

**Key terms**

- **F1 score**: a common metric for measuring classification accuracy (higher scores are better).
- **reshape()**: a method for changing the shape of data to prepare it for ML predictions.
- **r-squared**: a metric used to measure model performance for regressions (between 0 and 1, the higher the better)
- **sum of squared errors (SSE)**: a metric used to measure model performance for regressions: (0 or higher, the lower the better)
- **raw comparison test**: a test for model drift that consists of checking whether a new model score is worse than all previous scores
- **parametric significance test**: a test model drift that consists of checking whether a new model score is more than 2 standard deviations worse than the mean of all previous scores
- **non-parametric outlier test**: a test for model drift that consists of checking whether a new score is more than 1.5 interquartile ranges worse than the 25th or 75th percentile of previous scores
- **standard deviation**: a measure of how far spread apart the observations in a dataset are
- **interquartile range**: the difference between the 75th percentile and the 25th percentile of a set of observations
- **p-value**: a numerical result from a t-test used to determine whether two sets of numbers differ
- **t-test**: a statistical test for comparing two sets of numbers
- **statistical significance**: a concept describing the degree of evidence that two sets differ

- **latency**: Latency refers to the time delay in a program or the amount of time one part of your program has to wait for another part. If your processes take a long time to execute, it can cause latency in your project, and this could cause problems.
- **timestamp**: a timestamp is a representation of a specific date and time in a standard format. Modules related to time and timing often record timestamps to keep track of when processes begin and end.
- ``timeit``: the name of the module that we've used as a timer in this lesson.
- **integrity**: a dataset's state of being fully intact, with no missing or invalid entries
- **stability**: the similarity of data values between consecutive versions of datasets
- **dependencies**: 3rd-party modules that Python scripts import and depend on.
- **pip**: the Python package installer. You can use this tool from the workspace to install modules and check information about installed modules.
- **data imputation**: replacing missing entries with educated guesses about true values
- **mean imputation**: using column means to replace missing data entries


### Diagnosing and Fixing Operational Problems

**Key takeaways**

1. Time ML processes, and determine whether there are speed or latency issues
2. Check for integrity and stability issues in data
3. Check for dependencies, and resolve dependency issues
4. Perform data imputation: a method for resolving data integrity problems

**Key terms**

- **latency**: Latency refers to the time delay in a program or the amount of time one part of your program has to wait for another part. If your processes take a long time to execute, it can cause latency in your project, and this could cause problems.
- **timestamp**: a timestamp is a representation of a specific date and time in a standard format. Modules related to time and timing often record timestamps to keep track of when processes begin and end.
- ``timeit``: the name of the module that we've used as a timer in this lesson.
- **integrity**: a dataset's state of being fully intact, with no missing or invalid entries
- **stability**: the similarity of data values between consecutive versions of datasets
- **dependencies**: 3rd-party modules that Python scripts import and depend on.
- **pip**: the Python package installer. You can use this tool from the workspace to install modules and check information about installed modules.
- **data imputation**: replacing missing entries with educated guesses about true values
- **mean imputation**: using column means to replace missing data entries

### Model Reporting and Monitoring with API's

**Key takeaways**

1. **API configuration**: how to configure simple API's
2. **Endpoint scripting**: creating multiple, complex endpoints for API's
3. **Calling API's**: how to call API's in several ways
4. **Different API Methods**: different ways that API's can be called to provide or receive information or files

**Key terms**

- ``flask``: the module we used to create API's in Python
- **endpoint**: a specification for how API users interact with an API
- **host**: an IP address that specifies where an API will be hosted
- **port**: a number that users and API's both need to specify in order to interact
- **route**: the name by which a particular endpoint can be accessed by users
- **auxiliary function**: a function that helps the rest of a script accomplish its purpose
- **return statement**: a final line in a Python function that returns a specified value
- **localhost**: a special IP address, 127.0.0.1, that refers to the machine where the code is currently running.
- **query string**: a string that comes after an IP address or URL, and specifies arguments to be passed to an API
- **method**: a standard procedure for calling an API in a particular way
- **GET**: the default type of API call in Flask, used to obtain information from a project
- **POST**: a type of API call used to upload information or files to a project