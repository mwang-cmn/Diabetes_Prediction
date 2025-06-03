# Understanding Diabetes in Women: Signs, Risk Factors, and a Predictive AI Model

## Introduction
In medicine, diabetes mellitus is defined as a chronic metabolic disorder characterised by elevated levels of blood glucose (hyperglycaemia) resulting from defects in insulin secretion, insulin action, or both. The World Health Organization (WHO) and the American Diabetes Association (ADA) define diabetes based on the following diagnostic criteria (non-pregnant adults) [Source: WHO, ADA]:
#### Risk Factors for Diabetes
1. Type 1 Diabetes
An autoimmune form of diabetes in which the pancreas produces little or no insulin.

- Family history of type 1 diabetes (parent or sibling)
- Presence of diabetes-susceptibility genes (e.g. HLA variants)
- Younger age at onset (most diagnoses occur before age 14)
- Possible environmental triggers under investigation (e.g. viral infections, latitude effects)

2. Type 2 Diabetes
Characterized by insulin resistance and eventual insulin deficiency. Risk factors separate into non-modifiable and modifiable:

2.1 Non-modifiable
- Age over 45 years
- Family history of type 2 diabetes (first-degree relative)
- Certain ethnicities (African American, Hispanic, Native American, Asian American, Pacific Islander) 1
- History of gestational diabetes or prediabetes

2.2 Modifiable
- Overweight and obesity (BMI ≥ 25 kg/m²)
- Abdominal obesity (waist circumference > 89 cm in women)
- Physical inactivity
- Unhealthy diet (high in refined carbs, saturated fats)
- Hypertension and dyslipidemia (high blood pressure or cholesterol)

3. Prediabetes
A state of glucose dysregulation short of type 2 diabetes.
- Body mass index ≥ 25 kg/m² (≥ 23 kg/m² in Asian women)
- Waist circumference > 80 cm (increased central adiposity)
- Family history of type 2 diabetes
- History of gestational diabetes mellitus
- Impaired glucose tolerance or fasting glucose (IFG 100–125 mg/dL; IGT 140–199 mg/dL)
- Hypertension (≥ 130/85 mmHg) and dyslipidemia (triglycerides ≥ 150 mg/dL; HDL < 50 mg/dL)
- Physical inactivity
- Polycystic ovary syndrome (PCOS) and other endocrine disorders

4. Gestational Diabetes
Glucose intolerance first recognized during pregnancy.
- Overweight or obesity (pre-pregnancy BMI ≥ 25 kg/m²)
- Physical inactivity
- History of gestational diabetes in prior pregnancy
- Prediabetes or prior impaired glucose tolerance
- Polycystic ovary syndrome (PCOS)
- First-degree relative with diabetes
- Prior delivery of macrosomic infant (birth weight > 4.1 kg)
- High-risk ethnicity (Black, Hispanic, American Indian, Asian)

A recent global survey estimates that over 199 million women currently live with diabetes, a number projected to swell to 313 million by 2040. Diabetes is the ninth leading direct cause of death in women worldwide, accounting for 2.1 million deaths annually—many premature—and nearly half of women with diabetes remain unaware of their condition. Early, accurate classification between Type 1, Type 2, gestational and pre-diabetes can drive timely care and reduce complications from sight loss to cardiovascular diseases

## Problem Statement
**Objective**: 
The goal of this project is to build and evaluate multiclass classification models to accurately identify four types of diabetes in women i.e. Type 1, Type 2, Gestational Diabetes, and Prediabetes. I evaluated and compared the performance of four supervised learning algorithms:

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


1. **Random Forest** has the highest training accuracy (1.0000), suggesting it fits the training data exceptionally well. However, this could also indicate potential overfitting. Despite this, its validation accuracy (0.9854) and cross-validation accuracy (0.9818 ± 0.0018) remain strong and consistent, proving robust generalization.

2. **Gradient Boosting** shows high performance across all metrics with slightly lower validation and cross-validation accuracy than Random Forest (0.9844 and 0.9814 ± 0.0022, respectively). It may provide slightly better generalization depending on your dataset complexity.

3. **Quadratic Discriminant Analysis (QDA)** performs well but has slightly lower validation (0.9500) and cross-validation (0.9434 ± 0.0050) accuracy, making it a reasonable but less optimal choice.

4. **K-Nearest Neighbors (KNN)** performs significantly worse, with the lowest validation (0.7469) and cross-validation (0.7553 ± 0.0087) accuracy, making it an impractical choice for strong predictive performance.

## Recommendations
Based on the feature importance rankings from the **Random Forest model**, here are some key recommendations for **diabetes prevention and management among women**:

### **High-Impact Health Interventions:**
1. **Manage BMI (Most Important Factor)** – Maintain a healthy weight through balanced nutrition and regular exercise to reduce diabetes risk.
2. **Monitor Blood Glucose Levels** – Regular glucose screening helps detect early signs of diabetes or prediabetes.
3. **Control Cholesterol and Blood Pressure** – Healthy diet choices and medical interventions can prevent cardiovascular complications linked to diabetes.
4. **Track Waist Circumference** – A smaller waist circumference correlates with lower risk for insulin resistance and metabolic syndrome.
5. **Regulate Insulin Levels** – Lifestyle changes and medical guidance can help optimize insulin regulation.

### **Supportive Lifestyle Adjustments:**
6. **Nutrition & Pregnancy Care** – Manage weight gain during pregnancy to lower gestational diabetes risk.
7. **Increase Physical Activity** – Regular exercise helps control glucose and improves insulin sensitivity.
8. **Reduce Alcohol & Smoking** – Limiting consumption lowers metabolic risk factors and improves overall health.

### **Lower Impact But Still Relevant Considerations:**
9. **Genetic and Family History Awareness** – While genetics play a role, lifestyle modifications can still significantly alter risk.
10. **Socioeconomic and Ethnicity Factors** – Access to healthcare and education can improve diabetes prevention efforts.

