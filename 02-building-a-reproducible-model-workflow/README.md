# Building a Reproducible Model Workflow

## Lessons in this section:

- Introduction to Reproducible Model Workflows
- Machine Learning Pipelines
- Data Exploration and Preparation
- Data Validation
- Training, Validation and Experiment Tracking
- Final Pipeline, Release and Deploy
- **Project: Build an ML Pipeline for Short-term Rental Prices in NYC**

### Introduction to Reproducible Model Workflows

**Key takeaways**

1. An outline of the whole course
2. The prerequisites of the course
3. A brief intro to Machine Learning Operations (MLops)
4. Why reproducible model workflows and pipelines are important
5. Who the business stakeholders for MLops are
6. When to use reproducible workflows
7. The history of MLops
8. The tools and environment used throughout the course
9. A brief intro to the project you will build at the end of the course

### Machine Learning Pipelines

**Key takeaways**

1. The three levels of MLops
2. A refresher on using argparse
3. Versioning data and artifacts with Weights and Biases
4. ML pipeline Components in MLflow
5. Linking together the components of a pipeline
6. Conda vs. Docker

**Key terms**

- **Artifact**: The product of a pipeline component. It can be a file (an image, a model export, model weights, a text file...) or a directory.
- **Component**: One step in a Machine Learning Pipeline. In MLflow, a component is characterized by an environment file (``conda.yml`` if you are using conda), an entry point definition file (MLproject) and one or more scripts or commands to be executed and their supporting code.
- **Container**: A technology to package together the entire runtime of a software, i.e., the code itself and all its dependencies and data files. Containers can be spun up quickly, and they run identically across different environments.
- **Data Segregation**: The process of splitting the data, for example into train and test sets.
- **Environment (runtime)**: The environment where a software runs. In mlflow it is described by the ``conda.yml`` file (or the equivalent Dockerfile if using Docker).
- **Experiment**: A tracked and controlled execution of one or more related software components, or an entire pipeline. In W&B the experiment is called ``group``.
- **Hyperparameters**: The parameters of a model that are set by the user and do not vary during the optimization or fit. They cannot be estimated from the data.
- **Job Type**: Used by W&B to distinguish different components when organizing the ML pipeline. It is mostly used for the visualization of the pipeline.
- **Machine Learning Pipeline**: A sequence of one or more components linked together by artifacts, and controlled by hyperparameters and/or configurations. It should be tracked and reproducible.
- **Project**: All the code, the experiments and the data that are needed to reach a particular goal, for example, a classification of cats vs dogs.
- **Run**: The minimal unit of execution in W&B and in all tracking software. It usually represents the execution of one script or one notebook, but it can sometimes contain more; for example, one script that spawns other scripts.

### Data Exploration and Preparation

**Key takeaways**

1. What is and how to perform and track an Exploratory Data Analysis (EDA)
2. What is the pre-processing step, what should go into it and what shouldn't, and how to implement it in our ML pipeline
3. What is and how to perform the data segregation step (train/test split), and why it makes sense in many cases, provided we have enough data, to have multiple splits
4. What is feature engineering, the feature store and what problems it solves

**Key terms**

- **Exploratory Data Analysis (EDA)**: An interactive analysis performed at the beginning of the project, to explore the data and learn as much as possible from it. It informs many decisions about the development of a model. For example, we typically discover what kind of pre-processing the data needs before it can be used for training or for inference. It is also important to verify assumptions that have been made about the data and the problem.
- **Feature Engineering**: The process of creating new features by combining and/or transforming existing features.
- **Feature Store**: A MLops tool that can store the definition as well as the implementation of features, and serve them for online (real-time) inference with low latency and for offline (batch) inference with high throughput.


### Data Validation

**Key takeaways**

1. Write tests with pytest, both deterministic and non-deterministic
2. Use fixtures to share data between tests
3. Use ``conftest.py`` to add options to the command line of pytest so you can pass parameters and use it within components of ML pipelines

**Key terms**

- **Alternative Hypothesis**: In statistical hypothesis testing, the alternative hypothesis is a statement that contradicts the null hypothesis.
- **Deterministic Test**: A test that involves a measurement without randomness. For example, measuring the number of columns in a table.
- **ETL Pipelines**: Extract Transform Load pipelines. They are a classic structure for data pipelines. An ETL pipeline is used to fetch, preprocess and store a dataset.
- **Hypothesis Testing**: A statistical method to test a null hypothesis against an alternative hypothesis. The main element of HT is a statistical test.
- **Non-Deterministic Test**: A test that involves a measurement of a random variable, i.e., of a quantity with intrinsic randomness. Examples are the mean or standard deviation from a sample from a population. If you take two different samples, even from the same population, they will not have exactly the same mean and standard deviation. A non-deterministic test uses a statistical test to determine whether an assumption about the data is likely to have been violated.
- **Null Hypothesis**: In statistical hypothesis testing, the null hypothesis is the assumption that we want to test. For example, in case of the t-test the null hypothesis is that the two samples have the same mean.
- **P-Value**: The probability of measuring by chance a value for the Test Statistic equal or more extreme than the one observed in the data assuming that the null hypothesis is true.
- **Statistical Test**: An inference method to determine whether the observed data is likely or unlikely to occur if the null hypothesis is true. It typically requires the specification of an alternative hypothesis, so that a Test Statistic (TS) can be formulated and the expected distribution of TS under the null hypothesis can be derived. A statistical test is characterized by a false positive rate alpha (probability of Type I error) and a false negative rate beta (probability of a Type II error). There are many statistical tests available, depending on the null and the alternative hypothesis that we want to probe.
- **Test Statistic**: A random variable that can be computed from the data. The formula for the TS is specified by the appropriate statistical test that can be chosen once a null hypothesis and an alternative hypothesis have been formulated. For example, to test whether two samples have the same mean (null hypothesis) or a different mean (alternative hypothesis) we can use the t-test. The t-test specifies how to compute the TS appropriate for this case, as well as what is the expected distribution of TS under the null hypothesis.


### Training, Validation and Experiment Tracking

**Key takeaways**

1. The inference pipeline and the inference artifact: what they are and why are they used
2. How to perform an ordered experimentation phase, keeping track of and versioning data, code and hyper-parameters
3. How to create an inference pipeline with sklearn and export it with MLflow and scikit-learn (and a quick example with pytorch)
4. How to evaluate the inference artifact against the test dataset

**Key terms**

- **Experiment Tracking**: The process of recording all the necessary pieces of information needed to inspect and reproduce a run. We need to track the code and its version, the dependencies and their versions, all the metrics of interests, all the produced artifacts (images, model exports, etc.), as well as the environment where the experiment runs.
- **Hyperparameter Optimization**: The process of varying one or more hyperparameter of a run in order to optimize a metric of interest (for example, Accuracy or Mean Absolute Error).
- **Inference Artifact**: An instance of the Inference Pipeline containing a trained model.
- **Inference Pipeline**: A pipeline constituted of two steps: the pre-processing step and the model. The pre-processing step can be a pipeline on its own, and it manipulates the data and prepares them for the model. The inference pipeline should contain all the pre-processing that needs to happen during model development as well as during production. When the inference pipeline is trained (i.e., it contains a trained model) it can be exported to disk. The export product is called an Inference Artifact.

### Final Pipeline, Release and Deploy

**Key takeaways**

- How to bring everything together in a end-to-end ML pipeline
- How to release the code in GitHub, and how to assign version numbers using Semantic Versioning
- What is deployment, and how to deploy our inference artifact with MLflow and other tools for both online (real-time) and offline (batch) inference

**Key terms**

- **Release**: A static copy of the code that reflects the state at a particular point in time. It has a version attached to it, and a tag. The tag can be used to restore the repository (or a local copy of the code in the repository) to the state it was when the release was cut.
- **Semantic Versioning**: A common schema for versioning releases. A release version is made of 3 numbers, like 1.3.8, called respectively major, minor, and patch. The major number should be incremented for large, backward-incompatible changes. The minor number should be incremented when new features are added in a backward-compatible way. The patch number should be incremented for bug fixes and other small backward-compatible changes.
- **Deployment**: The operation of taking an inference artifact and putting it into production, so it can serve results to stakeholders and customers.