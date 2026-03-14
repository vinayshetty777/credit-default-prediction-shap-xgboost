# credit-default-prediction-shap-xgboost
Explainable Machine Learning for Credit Default Prediction Using SHAP on XGBoost Models

**Project Overview**
This project develops a machine learning model to predict credit card default risk using the XGBoost algorithm. In addition to prediction accuracy, the study focuses on model interpretability by applying SHAP (SHapley Additive exPlanations) to explain how individual features influence model predictions.

The goal is to build a reliable credit risk prediction model and provide transparent explanations for the decisions made by the machine learning system.

**Objectives**

● To determine the predictive power of the XGBoost model to predict credit defaults compared to the basis models of choice.
●	To identify the most significant characteristics of credit default predictions produced by the XGBoost model.
●	To determine the benefits of SHAP explanations to the transparency and interpretability of the XGBoost model.
●	To investigate how explainable machine learning can be applied in regulations and ethical decision-making in credit risk assessment.


**Dataset**

The dataset used in this project is the UCI Credit Card Default dataset.

File included in this repository:
UCI_Credit_Card (2).csv
The file can be directly accessed through this link "https://www.kaggle.com/code/bansodesandeep/credit-card-default-prediction/notebook"
**Repository Structure**
credit-default-prediction-shap-xgboost
│

├── README.md

├── UCI_Credit_Card (2).csv

├── Vinay_Analysis.ipynb

└── .gitignore
**Technologies Used**
- [Python](https://www.python.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [SHAP](https://shap.readthedocs.io/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Jupyter Notebook](https://jupyter.org/)
- [Google Colab](https://colab.research.google.com/)

## Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository

First, clone this repository to your local system:

```
git clone https://github.com/vinayshetty777/credit-default-prediction-shap-xgboost.git
```

Navigate into the project folder:

```
cd credit-default-prediction-shap-xgboost
```

---

### 2. Install Required Libraries

Install the required Python packages:

```
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn jupyter
```

---

### 3. Open the Notebook

Launch Jupyter Notebook:

```
jupyter notebook
```

Then open the file:

```
Vinay_Analysis.ipynb
```

---

### 4. Run the Analysis

Run all cells in the notebook to:

* Load and preprocess the dataset
* Train the XGBoost model
* Evaluate model performance
* Generate SHAP explanations for model predictions

---

### 5. Dataset

The dataset used in this project is included in the repository:

```
UCI_Credit_Card (2).csv
```

Make sure the dataset file is in the same directory as the notebook before running the analysis.

---

### Alternative: Run in Google Colab

You can also upload the notebook (`Vinay_Analysis.ipynb`) to **Google Colab** and run the project directly in the cloud without installing any packages locally.

**Model Explainability with SHAP**
SHAP (SHapley Additive exPlanations) is used to interpret the predictions of the XGBoost model. It provides insight into how each feature contributes to the prediction outcome.

SHAP helps to:

Understand feature importance

Explain individual predictions

Improve transparency of machine learning models
**Author**

Vinay Shetty
