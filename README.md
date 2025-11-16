# AIML-Task-2
# Titanic Survival Analysis: Preprocessing & EDA

This repository documents the process of data cleaning, feature engineering, and exploratory data analysis (EDA) performed on the Kaggle Titanic dataset. The goal is to prepare the data for a machine learning model and to uncover patterns related to passenger survival.

## 1. Data Cleaning & Preprocessing

The initial `Titanic-Dataset.csv` file was processed using the `titanic_preprocessing.py` script. The following actions were taken:

* **Missing Values**:
    * `Embarked`: 2 missing values were filled with the mode ('S').
    * `Cabin`: Dropped entirely due to >77% missing data.
    * `Age`: Left to be imputed *after* a train-test split (to prevent data leakage).
* **Feature Dropping**:
    * `PassengerId` and `Ticket` were dropped as they are non-predictive identifiers.
* **Feature Engineering**:
    * `Title`: Extracted from the `Name` column (e.g., "Mr", "Mrs", "Miss", "Master"). Rare titles were grouped into "Other".
    * `FamilySize`: Created by summing `SibSp` + `Parch` + 1.
    * `IsAlone`: A binary feature (1 if `FamilySize` == 1, 0 otherwise).
* **Categorical Encoding**:
    * `Sex`: Encoded using `LabelEncoder` (male: 1, female: 0).
    * `Embarked` & `Title`: Encoded using one-hot encoding.

## 2. Exploratory Data Analysis (EDA) - Key Findings

Analysis of the raw and cleaned data revealed several critical patterns.

### Finding 1: Survival is Not Random

The `Survived` vs. `Sex` vs. `Pclass` plot shows a clear and dramatic story. A passenger's chance of survival was heavily dependent on their **`Sex`** and **`Class`**.

* **Women and children first:** Females had a *much* higher survival rate than males across all classes.
* **Privilege mattered:** 1st class passengers had a much higher survival rate than 2nd class, who in turn had a higher rate than 3rd class.

### Finding 2: Engineered Features are Highly Predictive

The features we created (`Title`, `IsAlone`) are even stronger predictors than the original ones.

* **`Title` is a powerful proxy:** The `Title` feature is a brilliant combination of `Sex` and `Age`.
    * "Mr" had a very low survival rate (~16%).
    * "Miss" and "Mrs" had very high survival rates (~70-79%).
    * "Master" (young boys) had a high survival rate (~58%), indicating that "child" was a more important factor than "male" for this group.
* **Family vs. Alone**:
    * Passengers who were `IsAlone` had a lower survival rate (~30%) than those with family.
    * Small families (2-4 members) had the *best* survival rate. Very large families (>4) had a poor survival rate.

### Finding 3: Correlation & Model Prep

The correlation heatmap provides a final guide for modeling:

* **Strong Predictors**: `Sex`, `Pclass`, `Title_Mr`, `Title_Miss`, and `Title_Mrs` are all strongly correlated with `Survived` and will be excellent features.
* **Multicollinearity**:
    * `Sex` and `Title_Mr` are highly correlated.
    * `IsAlone` and `FamilySize` are perfectly correlated (we must drop one).
    * `Pclass` and `Fare` are strongly correlated.
* **Conclusion**: For a simple model (like Logistic Regression), we should drop `FamilySize`, `Name`, `SibSp`, and `Parch` to avoid multicollinearity. For a tree-based model (like a Random Forest), this is less of an issue.
