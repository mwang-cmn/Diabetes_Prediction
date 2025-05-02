# Understanding Diabetes in Women: Signs, Risk Factors, and a Predictive AI Model

## Introduction
In medicine, diabetes mellitus is defined as a chronic metabolic disorder characterised by elevated levels of blood glucose (hyperglycaemia) resulting from defects in insulin secretion, insulin action, or both. The World Health Organization (WHO) and the American Diabetes Association (ADA) define diabetes based on the following diagnostic criteria (non-pregnant adults) [Source: WHO, ADA]:

Diagnostic Criteria for Diabetes
A diagnosis is made if any of the following are true (preferably confirmed on a second test):

1. Fasting Plasma Glucose (FPG):
≥ 7.0 mmol/L (126 mg/dL) after no caloric intake for at least 8 hours

2. 2-hour Plasma Glucose (OGTT):
≥ 11.1 mmol/L (200 mg/dL) during an oral glucose tolerance test (75g)

3. HbA1c (Glycated Haemoglobin):
≥ 6.5% (48 mmol/mol)

4. Random Plasma Glucose:
≥ 11.1 mmol/L (200 mg/dL) with classic symptoms (polyuria, polydipsia, unexplained weight loss)

Diabetes is classified into:

- Type 1 Diabetes: Autoimmune destruction of insulin-producing beta cells

- Type 2 Diabetes: Insulin resistance combined with beta-cell dysfunction

- Gestational Diabetes: Hyperglycaemia diagnosed during pregnancy

- Other specific types: Includes monogenic diabetes, diseases of the exocrine pancreas, or drug-induced diabetes

A recent global survey estimates that over 199 million women currently live with diabetes, a number projected to swell to 313 million by 2040. Diabetes is the ninth leading direct cause of death in women worldwide, accounting for 2.1 million deaths annually—many premature—and nearly half of women with diabetes remain unaware of their condition. Early, accurate classification between Type 1, Type 2, gestational and pre-diabetes can drive timely care and reduce complications from sight loss to cardiovascular diseases

## Problem Statement
**Objective**: The goal of this project is to build and evaluate multiclass classification models to accurately identify four types of diabetes in women i.e. Type 1, Type 2, Gestational Diabetes, and Prediabetes. I evaluated and compared the performance of four supervised learning algorithms:

1. Random Forest Classifier

2. Gradient Boosting Classifier

3. K-Nearest Neighbors (KNN)

4. Quadratic Discriminant Analysis (QDA)

To ensure robustness and generalisability, **Stratified K-Fold Cross-Validation** (with n_splits=5) is used during model evaluation. The models are assessed using standard classification metrics including accuracy, precision, recall, and F1-score, with a focus on predictive performance. 
- **Scope**: The project focuses entirely on women


## Data Profile

| Column Name                     | Type      | Description                                                    | Notes                                     |
|---------------------------------|-----------|----------------------------------------------------------------|-------------------------------------------|
| Genetic Markers                 | object    | Presence of known diabetes-related genetic variants            | Encoded as dummy variables                |
| Family History                  | object    | Self-reported diabetes in first-degree relatives               | Binary (“Yes”/“No”), dummied              |
| Insulin Levels                  | int64     | Fasting insulin (µIU/mL)                                       | Continuous                                |
| Blood Glucose Levels            | int64     | Fasting blood glucose (mg/dL)                                  | Continuous                                |
| Glucose Tolerance Test          | object    | Result of 2-hour OGTT (“Normal”/“Impaired”/“Diabetic”)         | Categorical; dummy-encoded                |
| BMI                             | int64     | Body mass index (kg/m²)                                        | Will be binned using WHO female categories|
| Waist Circumference             | int64     | Waist measurement (cm)                                         | Continuous; ethnicity-adjusted thresholds |
| Physical Activity               | object    | Self-reported activity level (“Low”/“Moderate”/“High”)         | Dummy-encoded                             |
| Dietary Habits                  | object    | Diet quality (“Healthy”/“Unhealthy”)                           | Dummy-encoded                             |
| Smoking Status                  | object    | Tobacco use (“Never”/“Former”/“Current”)                       | Dummy-encoded                             |
| Alcohol Consumption             | object    | Drinking habits (“None”/“Moderate”/“Heavy”)                    | Dummy-encoded                             |
| Blood Pressure                  | int64     | Systolic blood pressure (mmHg)                                 | Continuous                                |
| Cholesterol Levels              | int64     | Total cholesterol (mg/dL)                                      | Continuous                                |
| Liver Function Tests            | object    | Liver enzyme normality (“Normal”/“Abnormal”)                   | Dummy-encoded                             |
| Previous Gestational Diabetes   | object    | History of gestational diabetes (“Yes”/“No”)                   | Dummy-encoded                             |
| Pregnancy History               | object    | Number of prior pregnancies or normality (“Normal”/“Complicated”) | Dummy-encoded                          |
| Weight Gain During Pregnancy    | int64     | Pregnancy weight gain (kg)                                     | Continuous                                |
| Ethnicity                       | object    | Self-reported ethnic group                                     | Dummy-encoded; used to adjust BMI/waist   |
| Socioeconomic Factors           | object    | Education/income bracket                                       | Dummy-encoded                             |
| Target                          | object    | Diabetes type (“Type1”/“Type2”/“Gestational”/“Pre-diabetic”)  | Multiclass label                          |

## Data Cleaning

There were no missing values in either the test or train datasets used in this project
  
- Convert categorical columns to dummies, dropping the first level to avoid multicollinearity.  
- Bin **BMI** into WHO female categories.   

## Exploratory Data Analysis (EDA)
  
- Distribution of each numerical feature by diabetes type  
- Boxplots of BMI, glucose, insulin across classes  
- Frequency of encoded lifestyle factors by class  

## Feature Selection & Engineering
- **SelectFromModel** on tuned Random Forest: threshold on feature importance to drop low-impact features.     
- Binned variables: WHO BMI categories, blood glucose ranges, blood pressure stages.  

## Model Evaluation
### Cross-Validation Setup
- **StratifiedKFold** with `n_splits=5`, `shuffle=True`, `random_state=42`.

### Models
1. **Gradient Boosting Classifier**  
2. **Random Forest Classifier**  
3. **K-Nearest Neighbors**  
4. **Quadratic Discriminant Analysis**

### Metrics
- Training accuracy  
- Cross-validation accuracy 
- Validation set accuracy  

## Results

| Model                          | Training Accuracy | Validation Accuracy | Cross-Validation Accuracy (± Std) |
|-------------------------------|-------------------|---------------------|-----------------------------------|
| Gradient Boosting             | 0.9820            | 0.9844              | 0.9814 ± 0.0022                   |
| Random Forest                 | 1.0000            | 0.9854              | 0.9818 ± 0.0018                   |
| K-Nearest Neighbors           | 0.8351            | 0.7469              | 0.7553 ± 0.0087                   |
| Quadratic Discriminant Analysis | 0.9458          | 0.9500              | 0.9434 ± 0.0050                   |

### Example Evaluation Code
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    print(f"{name}: CV accuracy {scores.mean():.3f} ± {scores.std():.3f}")
