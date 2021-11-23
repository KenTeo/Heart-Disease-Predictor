# Heart disease predictor and Prescriptive recommendation
#### 1. *Create a model to predict the likelihood of a person having heart Disease.*
#### 2. *Provide recommendation for patients to reduce heart disease risk.*

See the project deployed in the following website: 
https://heart-disease-advisor.herokuapp.com/
---
# Table of contents
- [Details of dataset](#dataset)
- [Extract and format the data](#extract)
- [Cleanse and wrangle the data](#clean)
- [Perform exploratory data analysis on the data](#eda)
- [Prepare the data for modelling](#prepare)
- [Modelling of data and optimization](#model)
- [Predict the likelihood of a heart disease and provide recommendation](#predict)
- [Details of website](#website)
---

<a name="dataset"></a>
## Details of dataset
#### The following are the data collected from 918 individuals:
>The heart failure prediction dataset was created by combining different datasets. 
>In this dataset, 5 heart datasets are combined over 11 common features. 

>The five datasets used for its curation are:
>Cleveland: 303 observations
>Hungarian: 294 observations
>Switzerland: 123 observations
>Long Beach VA: 200 observations
>Stalog (Heart) Data Set: 270 observations

>Total: 1190 observations
>Duplicated: 272 observations
>Final dataset: 918 observations

>Every dataset used can be found under the Index of heart disease datasets from UCI Machine Learning 
>Repository in the URL: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/
>Dataset downloaded from: https://www.kaggle.com/fedesoriano/heart-failure-prediction

**11 Common features:**
1. **Age**: age of the patient [years]
2. **Sex**: sex of the patient [M: Male, F: Female]
3. **ChestPainType**: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
4. **RestingBP**: resting blood pressure [mm Hg]
5. **Cholesterol**: serum cholesterol [mm/dl]
6. **FastingBS**: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
7. **RestingECG**: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or     ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes'       criteria]
8. **MaxHR**: maximum heart rate achieved [Numeric value between 60 and 202]
9. **ExerciseAngina**: exercise-induced angina [Y: Yes, N: No]
10. **Oldpeak**: oldpeak = ST [Numeric value measured in depression]
11. **ST_Slope**: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
12. **HeartDisease**: output class [1: heart disease, 0: Normal]

**Variables 1~11 are inputs while variable 12 will be the output.**
___
<a name="extract"></a>
## Extract and format the data 
*For the ease of extracting and formating the data(CSV) to dataframe, pandas is used*
```python
import pandas as pd
df = pd.read_csv("heart.csv")
```
___
<a name="clean"></a>
## Clean and wrangle the data
In the following code,  check for Null values in the data:
```python
df.info()
```
Data columns (total 12 columns):
| # | Column         | Non-Null Count | Dtype | 
|---|  ------        | -------------- | ----- | 
| 0 |  Age           |  918 non-null|    int64  
| 1 | Sex            | 918 non-null|    object 
| 2 |  ChestPainType |  918 non-null|    object 
| 3 |  RestingBP     |  918 non-null|    int64  
| 4 |  Cholesterol   |  918 non-null|    int64  
| 5 |  FastingBS     |  918 non-null|    int64  
| 6 |  RestingECG    |  918 non-null|    object 
| 7 |  MaxHR         |  918 non-null|    int64  
| 8 |  ExerciseAngina|  918 non-null|    object 
| 9 |  Oldpeak       |  918 non-null|    float64
| 10 | ST_Slope      |  918 non-null|    object 
| 11 | HeartDisease  |  918 non-null|    int64  

From result above, there are no NULL values in the data.
___
<a name="eda"></a>
## Exploratory Data Analysis.
The summary of the key findings from the data.
#### Numerical variables
![image](https://drive.google.com/uc?export=view&id=1Pnro2-DO441NzYw0YjpLCJ7lSFHOVFBe)

From the correlation heatmap above, we can see that for numerical variables, maximum heart rate and old peak has the highest correlation to Heart disease.
#### Categorical variables
**Histogram plot of Chest Pain Type**
![image](https://drive.google.com/uc?export=view&id=16bzYtit6kGNGmVSxN6K9l80b3Ysb6HHc)

|                                   | ATA | NAP | ASY | TA |
|-----------------------------------|-----|-----|-----|----|
|Probability of having Heart Disease|14%  |35%  |79%  |43% |

**Histogram plot of Exercise Angina**
![image](https://drive.google.com/uc?export=view&id=1f3X5RDh3w1-bY9Oz9hDhyyUJ0jJD-Ush)

|                                   | No Exercise Angina | Have Exercise Angina |
|-----------------------------------|--------------------|----------------------|
|Probability of having Heart Disease|35%                 |85%                   |

**Histogram plot of ST Slope**
![image](https://drive.google.com/uc?export=view&id=1beHArcPb0L57GFUTT16KwLQhO2rOhaqq)

|                                   | Up | Flat | Down |
|-----------------------------------|-----|-----|-----|
|Probability of having Heart Disease|20%  |83%  |78%  |

___
<a name="prepare"></a>
## Prepare the data for modelling
*Change Sex and ExeriseAngina variables to binary input. Change "ChestPainType", "RestingECG" and "ST_Slope"
categorical variables to one vs rest binary variables using get_dummies function.*
```python
df.replace({"Sex":{"M" : 1, "F" : 0}}, inplace=True)
df.replace({"ExerciseAngina":{"Y" : 1, "N" : 0}}, inplace=True)
train = pd.get_dummies(df, columns={"ChestPainType", "RestingECG", "ST_Slope"})
```
___
<a name="model"></a>
## Modelling of data and optimization
#### Modelling of data
6 Different types of algorithm is tested and the algorithm with the highest accuracy score is chosen as the final model.
Accuracy score is a good judging criteria as the data is not skewed but relatively evenly split. With 508 samples with Heart disease and 410 samples without heart disease.

The following is the pre-processing for the data:
- 10-fold cross validation is used to obtain a reliable accuracy score from the model.
- logistic regression and support vector machine is scaled to improve the convergence of the model.
- One Hot encoder is used on categorical variables.
- Pipeline is build for Scaling and One Hot encoder with the classifier to prevent data leakage.

The result or the testing of the algorithem with 10-fold cross-validation is shown below.
![image](https://drive.google.com/uc?export=view&id=1a0zdgjCAFjaMBxSWQE_2jOl0jWHzCpry)
From the table above, XGBoost has the highest accuracy score(highest score mean). Hence, it will be chosen as the final model. 

*Note: As the dataset is very small(918 samples) we will not be able to see full the power of the XGBoost algorithm but in a actual project where we are able to collect more data. The accuracy score of the XGBoost algorithm will be much higher than other algorithm as shown above.

#### Optimization of Model
RandomizedSearchCV is applied first to randomly search for the rough parameters where the model has the highest accuracy score. Next, GridSearchCV is used to search every possible combination of the parameters in the region where the accuracy score for the model is the highest during the RandomizedSearchCV process to fine tune the parameters.
___
<a name="predict"></a>
## Predicting of data and providing recommendation
Using the predict_proba() function, we are able to obtain the predicted probability of having heart disease.
The likelihood of heart disease for the new sample is broken down into 5 groups.
They are very low risk(0~0.2), low risk(0.2~0.4), medium risk(0.4~0.6), high risk(0.6~0.8) and very high risk(0.8~1).
The numbers from 0~1 represents the probability output of the XGBoost algorithm where a number closer to 1 means higher probability that there is heart disease and 0 is likely to have no heart disease.
![image](https://drive.google.com/uc?export=view&id=1_6IjQJSFxve8ff-Sm4hjLqdv-FEOKUVf)

#### Recommendation for User
In order for users of the predictor to reduce their risk of a heart disease. 
They can work on the following variables which can be improve:
4. RestingBP: resting blood pressure [mm Hg]
5. Cholesterol: serum cholesterol [mm/dl]
6. FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
___

<a name="website"></a>
## Details of website
#### Website structure
```
website
|---Procfile
|---wsgi.py
|---requirements.txt
|---app.py
|---model.bin
|---templates
|      |-------home.html
|      |-------predict.html
|---static
|      |-------styles.css
|      |-------<images>
```
#### Link for project: https://heart-disease-advisor.herokuapp.com/
### Main page of website (home.html)
![image](https://drive.google.com/uc?export=view&id=1G3xfiFLB5o9eBUMFiH5m83Ws-Nb1gCfT)
### Prediction page of website (predict.html) (Example of a possible output)
![image](https://drive.google.com/uc?export=view&id=1BLf51a7uG-Uh8akdZYSwJ1m7I_rllilZ)

---
## The End
