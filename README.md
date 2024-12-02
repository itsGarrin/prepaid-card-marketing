# Prepaid Card Marketing Analysis

This repository contains a data analysis and machine learning project focused on the CFPB financial wellness dataset. The primary objective is to identify characteristics of applicants likely to use prepaid cards, helping guide targeted marketing strategies.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methods](#methods)
4. [Results](#results)
5. [Dependencies](#dependencies)
6. [How to Run](#how-to-run)
7. [Contributing](#contributing)
8. [License](#license)

---

## Introduction

Prepaid cards are an essential tool for individuals managing finances without traditional banking access. Understanding the demographics and characteristics of individuals more likely to use prepaid cards can provide valuable insights for companies targeting these consumers. This project leverages data analysis and machine learning techniques to uncover such patterns.

---

## Dataset

The dataset used in this project is the **CFPB Financial Wellness Dataset**:
- **Features**: 20 demographic and financial wellness metrics.
- **Observations**: 6000 data points.
- **Target**: Binary classification indicating prepaid card use.

The dataset is highly imbalanced, with a minority class representing prepaid card users.

---

## Methods

### Data Preprocessing
- Handling missing values.
- Feature scaling and encoding.
- Addressing class imbalance using SMOTE and class-weight adjustments.

### Exploratory Data Analysis (EDA)
- Identifying trends and correlations within the data.
- Visualizing distributions, relationships, and class imbalance.

### Machine Learning Models
The following models were evaluated for predictive performance:
1. Logistic Regression
2. Random Forest
3. AdaBoost
4. XGBoost

Metrics such as accuracy, precision, recall, F1-score, and AUC-ROC were used for evaluation.

---

## Results

The project highlights:
- Significant features that influence prepaid card usage.
- Model performance in predicting the minority class.

---

## Dependencies

The project was developed using Python. Below are the key dependencies:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- XGBoost
