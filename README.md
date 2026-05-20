# HR-Employee-Prediction-using-Machine-Learning
HR Employee Attrition Prediction Using Machine Learning
Overview

Employee attrition is one of the major challenges faced by organizations because high employee turnover increases recruitment costs, training expenses, workload pressure, and productivity loss. Predicting employee attrition helps Human Resource (HR) departments identify employees at risk of leaving and implement retention strategies proactively.

This project focuses on predicting employee attrition using machine learning techniques on the IBM HR Analytics Employee Attrition dataset. Multiple machine learning models were developed and evaluated to determine the most effective approach for identifying employees likely to leave the organization.

The project includes:

Data Cleaning and Preprocessing
Exploratory Data Analysis (EDA)
Handling Class Imbalance using Oversampling
Logistic Regression
Decision Tree Classification
Random Forest Classification
Threshold Tuning
ROC Curve and AUC Analysis
10-Fold Cross Validation
Model Comparison and Business Insights
Problem Statement

Organizations experience significant financial and operational challenges due to employee attrition. Replacing employees involves:

Recruitment costs
Training expenses
Knowledge loss
Reduced productivity
Increased workload on existing employees

The objective of this project is to build predictive machine learning models capable of identifying employees who are likely to leave the organization.

Project Objectives

The main objectives of this project are:

Analyze employee-related factors influencing attrition
Perform Exploratory Data Analysis (EDA)
Handle class imbalance in the dataset
Build multiple machine learning classification models
Compare model performance using evaluation metrics
Identify the best model for HR attrition prediction
Generate business insights for employee retention
Dataset Information
Dataset

IBM HR Analytics Employee Attrition Dataset

Dataset Characteristics
Attribute	Value
Number of Rows	1470
Number of Columns	35
Target Variable	Attrition
Features Included

The dataset contains:

Employee demographic information
Work-related variables
Compensation details
Satisfaction metrics
Experience-related variables

Examples:

Age
Gender
MonthlyIncome
Department
JobRole
OverTime
JobSatisfaction
WorkLifeBalance
YearsAtCompany
EnvironmentSatisfaction
Technologies and Libraries Used
Programming Language
R Programming
Libraries
tidyverse
caret
randomForest
ROCR
corrplot
ggplot2
rpart
rpart.plot
scales
gridExtra
Project Workflow
1. Data Loading

The IBM HR Employee Attrition dataset was imported into R for analysis.

2. Data Cleaning and Preprocessing

The following preprocessing steps were performed:

Removed Unnecessary Columns

The following columns were removed because they did not provide predictive value:

EmployeeNumber
EmployeeCount
Over18
StandardHours
Converted Variables
Categorical variables were converted into factors
Attrition target variable was converted into binary format
One-Hot Encoding

Categorical variables were transformed into numerical format using:

model.matrix()

This created machine-learning-ready predictor variables.

Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to identify important trends and relationships associated with employee attrition.

Key EDA Findings
1. Attrition Distribution
The dataset was imbalanced
Majority of employees belonged to the “No Attrition” category
2. Overtime and Attrition
Employees working overtime showed significantly higher attrition rates
3. Monthly Income and Attrition
Employees with lower income were more likely to leave
4. Job Satisfaction and Attrition
Lower satisfaction levels were strongly associated with attrition
5. Age and Attrition
Younger employees experienced higher attrition rates
6. Correlation Analysis

Strong correlations were observed among variables such as:

MonthlyIncome
JobLevel
TotalWorkingYears
Handling Class Imbalance

The dataset contained significantly fewer attrition cases compared to non-attrition cases.

To address this issue:

Random Oversampling was applied to the minority class
Minority class samples were duplicated until both classes were balanced

This improved the model’s ability to learn attrition patterns.

Machine Learning Models

Three classification models were implemented and compared.

1. Logistic Regression

Logistic Regression was selected because:

it is highly interpretable,
effective for binary classification,
provides probability-based predictions.
Logistic Regression Performance
Metric	Value
Accuracy	76.82%
Precision	39.46%
Recall	81.69%
F1-Score	53.21%
AUC	0.8336
2. Decision Tree

Decision Trees were implemented to:

identify hierarchical decision rules,
provide business interpretability.
Decision Tree Performance
Metric	Value
Accuracy	72.73%
Precision	31.58%
Recall	59.15%
F1-Score	41.18%
AUC	0.6890
3. Random Forest

Random Forest was implemented as an ensemble learning method using multiple decision trees.

Random Forest Performance
Metric	Value
Accuracy	85.23%
Precision	63.64%
Recall	19.72%
F1-Score	30.11%
AUC	0.8340

Although Random Forest achieved the highest Accuracy, its Recall was very low, meaning it failed to identify many actual attrition cases.

Why Recall Was Prioritized

The primary business objective was:

identify employees who are likely to leave the organization.

In HR analytics:

Missing actual attrition cases is costly
False positives are more acceptable

Therefore:

Recall was considered more important than Accuracy.
Threshold Tuning

Threshold tuning was performed on the Logistic Regression model by evaluating thresholds from:

0.10 to 0.90

The final threshold selected was:

0.50

This threshold provided:

High Recall
Balanced F1-Score
Strong overall business usefulness
ROC Curve and AUC Analysis

ROC curves were generated to compare model classification performance across different thresholds.

AUC Interpretation
AUC Range	Interpretation
0.9+	Excellent
0.8 – 0.9	Very Good
0.7 – 0.8	Acceptable
0.5	Random Guessing

Logistic Regression and Random Forest achieved strong AUC values above 0.83.

Cross Validation

10-Fold Cross Validation was performed on the Logistic Regression model to:

evaluate robustness,
reduce overfitting,
ensure generalization capability.
Cross Validation Results
Metric	Value
Accuracy	77.43%
Kappa	0.5487

The cross-validation accuracy was very close to testing accuracy, indicating stable model performance.

Final Model Selection

Logistic Regression was selected as the final model because:

it achieved the highest Recall,
maintained strong AUC performance,
produced balanced overall classification performance,
aligned best with the business objective.
Key Business Insights

The project identified several important factors associated with employee attrition:

Overtime strongly increased attrition risk
Lower monthly income increased turnover probability
Low job satisfaction contributed to attrition
Younger employees showed higher attrition rates
Poor work-life balance influenced employee turnover

These insights can help HR departments develop targeted retention strategies.

Future Improvements

Potential future improvements include:

SMOTE-based balancing techniques
Hyperparameter tuning
XGBoost implementation
Feature selection methods
Larger datasets
Deep learning approaches
Repository Structure
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── scripts/
│   └── hr_attrition_prediction.R
│
├── outputs/
│   ├── plots/
│   ├── confusion_matrices/
│   └── roc_curves/
│
├── report/
│   └── HR_Employee_Attrition_Report.pdf
│
├── presentation/
│   └── HR_Employee_Attrition_Presentation.pptx
│
└── README.md
Conclusion

This project successfully demonstrated how machine learning can be used to predict employee attrition and support HR decision-making.

Among all evaluated models:

Logistic Regression achieved the best business-focused performance
High Recall made it effective for identifying employees likely to leave
The analysis provided actionable HR insights for improving employee retention

The project highlights the importance of selecting evaluation metrics based on business objectives rather than relying solely on Accuracy.

References
IBM HR Analytics Employee Attrition Dataset
Kuhn, M. Caret Package Documentation
Breiman, L. Random Forests, Machine Learning Journal
R Documentation for Logistic Regression, Decision Trees, and Random Forest Algorithms

Author
Team:
Rithik Rathinavel Ragupathi
Irfan Saleemudeen
Sashank Addanki Venkata Naga
Lakshmi Srujana Sushma Pedapati
