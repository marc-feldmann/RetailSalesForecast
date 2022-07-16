from unittest import TestResult
import pandas as pd
import numpy as np
import numpy.ma as ma
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import math
import graphviz

# NOTES or project presentation (e.g. blog, CV, ...):
# emphasize that focus here, since dataset was mostly given, was on modelling, i.e. testing out and optimizing XGBoost.
# However, make sure to keep write-up tone authoritative, rather than explorative. 
# also emphasize: speciality of time-series prediction: seasonality! autocorrelation! means that preserving temporal patterns/'consecutiveness' (?)
# is very important as it hold predictive power - means e.g. that randomized Cross Validation is harmful

# END TO END GUIDANCE: https://www.youtube.com/watch?v=GrJP9FLV3FE
# STRATEGY:
# 1) GET BASIC XGBOOST MODEL TO RUN, compare to baseline: rule-of-thumb/heuristic forecast (prior year value + typical YOY-growth +/- extraordinary effects) 
# OPTIMIZATION vielleicht komplett für später aufheben: 80/20!! Siehe Winner Description, Feature Engineering hat nur marginal das Modell verbessert.
# Nur jetzt schon bzw. überhaupt in Optimierung gehen, wenn XGBoost den rule-of-thumb forecast (proxy für human store manager forecast) nicht knackt.
# 2) Optimize (if at all: only little bit for demonstrative purposes - DON'T OVERENGINEER!!!!!!!):
#   2a) focus on iterating between FEATURE ENGINEERING, FEATURE SELECTION, AND VALIDATION. - 'rhinking work' - "if I had all the data": in real life: what predicts store sales? gooGle! "need to understand what we are modelling in real life! business understanding!"
#   2b) then grid search over most important hyperparameters  #

# LOAD DATASET(S)
data_stores = pd.read_csv(filepath_or_buffer = "C:\\Users\\marc.feldmann\\Documents\\data_science_local\\RSP\\store.csv", delimiter = ",")
data_train = pd.read_csv(filepath_or_buffer = "C:\\Users\\marc.feldmann\\Documents\\data_science_local\\RSP\\train.csv", delimiter = ",")
## 'Sales' column missing in following, goal is to predict: 
data_test = pd.read_csv(filepath_or_buffer = "C:\\Users\\marc.feldmann\\Documents\\data_science_local\\RSP\\test.csv", delimiter = ",")

# EDA and CLEANING
## "read up on XGBoost / regression tree requirements (regarding e.g. data type, normalization, variable distribution, missing values)"
## XGBoost preprocessing requirements:
## (X) no nans (in training data output variables) 
## (X) numerical variables
## (X) dummy-encode categorical input variables (label encoding will mislead XGBoost because interprets as ordinal relationship!)
## (X) expects missing values to be represented as "0" (float type)
## (X) normalize target variable: if distribution is skewed, tree nodes/thresholds will be higher > better accuracy for extremve value predictions, but bad on main bulk of data!
##  no transformation of input variables required as decision tree models are insensitive to that


## EDA and CLEANING: training data
### get first impression of how data looks
data_train.info()
data_train[12267:12300]

for i in data_train.columns:
    print("Value distribution in column '", i, "':")
    round(data_train[i].value_counts(normalize=True, dropna=False), 2)
    print("Data type of column '", i, "':")
    data_train[i].dtype


### for all columns ('Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday'):
### > inspect and potentially clean: data type (XGBoost requires numerical data), missing values (impute with numerical value or replace with 0 [XGBoost requirement])
### > no need to worry about outliers in predictor variables since decision tree models (such as XGBoost) handle them  well (b/c DTs care primarily about splits, not distances)

### inspect column Store
min(data_train['Store']), max(data_train['Store']), data_train['Store'].isna().sum()
data_train['Store'].dtype

### inspect column DayOfWeek
round(data_train['DayOfWeek'].value_counts(normalize=True, dropna=False), 3)
data_train['DayOfWeek'].isna().sum()
data_train['DayOfWeek'].dtype

### inspect column Date
min(data_train['Date']), max(data_train['Date']), data_train['Date'].isna().sum()
data_train['Date'].dtype
data_train.Date = pd.to_datetime(data_train.Date)
#### To turn into int and properly capture predictive power, split up in three columns year, month, day which are all numerical types
data_train['Year'] = data_train['Date'].dt.year
data_train['Month'] = data_train['Date'].dt.month
data_train['Day'] = data_train['Date'].dt.day

### inspect column Sales
data_train['Sales'].describe().apply(lambda x: format(x, 'f'))
data_train['Sales'].isna().sum()
data_train['Sales'].dtype

f, ax = plt.subplots(2,2,figsize=(8,4))
sns.boxplot(data=data_train, y='Sales', showfliers=False, palette=['grey'], ax=ax[0][0])
ax[0, 0].set_ylabel('Daily Sales/Store (EUR)')
sns.distplot(data_train['Sales'], ax=ax[0, 1])
data_train.groupby([(data_train.Date.dt.year),(data_train.Date.dt.month)])['Sales'].sum().plot(ax=ax[1][0])
plt.show()
#### Insight 1: data description mentioned some stores were closed for refurbishment - assume for now that this explains Sales decline between May - Dec 14
#### Insight 2: prediction target variable Sales is bimodal and right-skewed: should be log transformed and zeroes excluded when training XGBoost model
  
### inspect column Customers
data_train['Customers'].describe().apply(lambda x: format(x, 'f'))
data_train['Customers'].isna().sum()
data_train['Customers'].dtype

f, ax = plt.subplots(2,2,figsize=(8,4))
sns.boxplot(data=data_train, y='Customers', showfliers=False, palette=['grey'], ax=ax[0][0])
ax[0, 0].set_ylabel('Daily Customers/Store')
sns.distplot(data_train['Customers'], ax=ax[0, 1])
ax[0, 1].set_xlim([0, 7000])
data_train.groupby([(data_train.Date.dt.year),(data_train.Date.dt.month)])['Customers'].sum().plot(ax=ax[1][0])
plt.show()

### inspect column Open
round(data_train['Open'].value_counts(normalize=True), 3)
data_train['Open'].isna().sum()
data_train['Open'].dtype

### inspect column Promo
round(data_train['Promo'].value_counts(normalize=True), 3)
data_train['Promo'].isna().sum()
data_train['Promo'].dtype

### inspect column StateHoliday
round(data_train['StateHoliday'].value_counts(normalize=True), 3)
data_train.loc[data_train['StateHoliday'] == '0', 'StateHoliday'] = 0
data_train['StateHoliday'].isna().sum()
data_train['StateHoliday'].dtype
data_train = pd.get_dummies(data_train, columns=['StateHoliday'])

### inspect column SchoolHoliday
round(data_train['SchoolHoliday'].value_counts(normalize=True), 3)
data_train['SchoolHoliday'].isna().sum()
data_train['SchoolHoliday'].dtype


## EDA and CLEANING: test data
### get first impression of how data looks
data_test.info()
data_test[12267:12300]

for i in data_test.columns:
    print("Value distribution in column '", i, "':")
    round(data_test[i].value_counts(normalize=True, dropna=False), 2)
    print("Data type of column '", i, "':")
    data_test[i].dtype


### for all columns ('Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday'):
### > inspect and potentially clean: data type (XGBoost requires numerical data), missing values (impute with numerical value or replace with 0 [XGBoost requirement])
### > no need to worry about outliers in predictor variables since decision tree models (such as XGBoost) handle them  well (b/c DTs care primarily about splits, not distances)

### inspect column Store
min(data_test['Store']), max(data_test['Store']), data_test['Store'].isna().sum()
data_test['Store'].dtype

### inspect column DayOfWeek
round(data_test['DayOfWeek'].value_counts(normalize=True, dropna=False), 3)
data_test['DayOfWeek'].isna().sum()
data_test['DayOfWeek'].dtype

### inspect column Date
min(data_test['Date']), max(data_test['Date']), data_test['Date'].isna().sum()
data_test['Date'].dtype
data_test.Date = pd.to_datetime(data_test.Date)
#### To properly capture predictive power, should engineer later: perhaps split up in three columns year, month, day which are all numerical types
data_test['Year'] = data_test['Date'].dt.year
data_test['Month'] = data_test['Date'].dt.month
data_test['Day'] = data_test['Date'].dt.day

### inspect column Sales
#### MISSING - is in the future so we do not now that yet - this should be predicted!

### inspect column Customers
#### MISSING - is in the future so we do not now that yet

### inspect column Open
round(data_test['Open'].value_counts(normalize=True), 3)
data_test['Open'].unique()
data_test['Open'].isna().sum()
data_test['Open'].dtype

### inspect column Promo
round(data_test['Promo'].value_counts(normalize=True), 3)
data_test['Promo'].isna().sum()
data_test['Promo'].dtype

### inspect column StateHoliday
round(data_test['StateHoliday'].value_counts(normalize=True), 3)
data_test.loc[data_test['StateHoliday'] == '0', 'StateHoliday'] = 0
data_test['StateHoliday'].isna().sum()
data_test['StateHoliday'].dtype
data_test = pd.get_dummies(data_test, columns=['StateHoliday'])

### inspect column SchoolHoliday
round(data_test['SchoolHoliday'].value_counts(normalize=True), 3)
data_test['SchoolHoliday'].isna().sum()
data_test['SchoolHoliday'].dtype



## EDA and CLEANING: store data
## store data
data_stores.info()
data_stores.head()

for i in data_stores.columns:
    print("Value distribution in column '", i, "':")
    round(data_stores[i].value_counts(normalize=True, dropna=False), 2)
    print("Data type of column '", i, "':")
    data_stores[i].dtype

### for all columns ('Store', 'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval'):
### > inspect and potentially clean: data type (XGBoost requires numerical data), missing values (impute with numerical value or replace with 0 [XGBoost requirement])
### > no need to worry about outliers in predictor variables since decision tree models (such as XGBoost) handle them  well (b/c DTs care primarily about splits, not distances)

### inspect column Store
min(data_stores['Store']), max(data_stores['Store']), data_stores['Store'].isna().sum(), data_stores['Store'].shape[0]
data_stores['Store'].dtype

### inspect column StoreType
round(data_stores['StoreType'].value_counts(normalize=True), 3)
data_stores['StoreType'].isna().sum()
data_stores = pd.get_dummies(data_stores, columns=['StoreType'])

### inspect column Assortment
round(data_stores['Assortment'].value_counts(normalize=True), 3)
data_stores['Assortment'].isna().sum()
data_stores = pd.get_dummies(data_stores, columns=['Assortment'])

### inspect column CompetitionDistance
data_stores['CompetitionDistance'].describe().apply(lambda x: format(x, 'f'))
data_stores['CompetitionDistance'].isna().sum()
data_stores['CompetitionDistance'].dtype

fig, ax = plt.subplots(2)
sns.boxplot(y=data_stores['CompetitionDistance'], showfliers=False, palette=['grey'], ax=ax[0])
ax[0].set_ylabel('CompetitionDistance/Store')
sns.distplot(data_stores['CompetitionDistance'], ax=ax[1])
ax[1].set_xlim([0, 7000])
plt.show()

### inspect column CompetitionOpenSinceYear
round(data_stores['CompetitionOpenSinceYear'].value_counts(normalize=True), 3)
data_stores['CompetitionOpenSinceYear'].isna().sum()
data_stores['CompetitionOpenSinceYear'].dtype

### inspect column CompetitionOpenSinceMonth
round(data_stores['CompetitionOpenSinceMonth'].value_counts(normalize=True), 3)
data_stores['CompetitionOpenSinceMonth'].isna().sum()
data_stores['CompetitionOpenSinceMonth'].dtype

### inspect column Promo2
round(data_stores['Promo2'].value_counts(normalize=True), 3)
data_stores['Promo2'].isna().sum()
data_stores['Promo2'].dtype

### inspect column Promo2SinceWeek
round(data_stores['Promo2SinceWeek'].value_counts(normalize=True), 3)
data_stores['Promo2SinceWeek'].isna().sum()
data_stores['Promo2SinceWeek'].dtype

### inspect column Promo2SinceYear
round(data_stores['Promo2SinceYear'].value_counts(normalize=True), 3)
data_stores['Promo2SinceYear'].isna().sum()
data_stores['Promo2SinceYear'].dtype

### inspect column PromoInterval
round(data_stores['PromoInterval'].value_counts(normalize=True), 3)
data_stores['PromoInterval'].isna().sum()
data_stores.loc[data_stores['PromoInterval'] == 'Jan,Apr,Jul,Oct', 'PromoInterval'] = 'JanStart_Quart'
data_stores.loc[data_stores['PromoInterval'] == 'Feb,May,Aug,Nov', 'PromoInterval'] = 'FebStart_Quart'
data_stores.loc[data_stores['PromoInterval'] == 'Mar,Jun,Sept,Dec', 'PromoInterval'] = 'MarStart_Quart'
data_stores = pd.get_dummies(data_stores, columns=['PromoInterval'], dummy_na=True)

## "we learned from EDA... implications..."
### initial observations about datasets:
### - Sales column in test missing - this is the prediction goal! (6 weeks - 41088 rows, as also indicate by sample upoad file)
### - Höchstwahrscheinlich findet die Ermittlung der Prognosegüte dann beim Upload statt
### - time spans:
### 	- training data: from 01.01.2013 to 31.07.2015
### 	- test data: from 01.08.2015 to 17.09.2015
### - I have on average around 1000 rows per each of the 1115 stores


# MERGE STORE DATA
data_train.info()
### ACHTUNG spalten der beiden Teildatensätze unterscheiden sich!!
data_train = pd.merge(data_train, data_stores, how='left')
data_test = pd.merge(data_test, data_stores, how='left')


# MODEL TRAINING
## (NOTE: If training and test data split had not been done already, and if we were building a classification instead of regression tree,
## we would here use 'stratified' splitting to maintain the 'balance' (a word for distribution of categorical variables)
## of the prediction target variable that exists in the overall data set also in the training and test sets)    

## HOLDOUT set creation:
### XGBoost is a machine learning model where learning happens via gradient descend: thus requires some
### unbiased data to be able to assess after each training iteration whether is is 'going' into the right direction (= 'learning'), that is, minimizing error
### To create holdout set, normally would randomly split data into training and test set. XGBoost can even do that automatically via its 'cv' parameter.
### However, we need to be mindful that we are working with time series data. Randomized test/train-splitting will most likely 
### compromise endogenous, temporal patterns existing in the data that are important for prediction.
### Will thus create holdout set manually, in followin manner: 
### Normally, since six consecutive weeks of data to predict, would use the same six weeks from the prior year as holdout set.
### However, this might, again, tear apart any possible temporal patterns that might have predictive power and might thus
### be important to catch in model training. Thus will instead used the six weeks directly before the predicted six weeks (in other words,
### the most recent six weeks in training set).

temp = data_train
data_holdout = temp.loc[(data_train['Date'] > '2015-06-18')]
data_train = temp.loc[(data_train['Date'] <= '2015-06-18')]
### check:
min(data_train['Date']), max(data_train['Date']), data_train['Date'].shape
min(data_holdout['Date']), max(data_holdout['Date']), data_test['Date'].shape

## MODEL evaluation: create RMSPE scoring function
## "predt" = Vektor mit den vom Modell produzierten (log-transformierten) Vorhersagen/predictions per Store per Day für das Holdout Set
## "dtrain" = data_holdout as DMatrix inklusive der tatsächlich observed target values (the actual 'right' values = "labels"); die Info was das "label" ist ist in der DMatrix als Metainfo gespeichert
# def rmspe(predt, dtrain):
#     "Root mean squared percetage error (RMSPE):"
#     labels = dtrain.get_label()
#     y = [math.exp(x)-1 for x in labels[labels > 0]]
#     yhat = [math.exp(x)-1 for x in predt[labels > 0]]
#     ssquare = [math.pow((y[i] - yhat[i])/y[i],2) for i in range(len(y))]
#     return 'RMSPE = ', math.sqrt(np.mean(ssquare))

def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w

# def rmspe(yhat, y):
#     w = ToWeight(y)
#     rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
#     return rmspe

def rmspe_xg(yhat, y):
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe


## Fitting XGBoost regression model:
#### it seem XGBoost is in some way integrated with sklearn: https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
# vstack to create proper dimensionality to feed into DMatrix
# log Sales both to normalize distribution (right-skewed) and prevent overflow errors in RMSPE expoential operations; log(x+1) to prevent log(0) infinity

data_train = data_train[data_train['Sales'] > 0]
data_holdout = data_holdout[data_holdout['Sales'] > 0]

X_train, y_train = data_train.drop(columns=['Sales', 'Date', 'Customers', 'StateHoliday_b', 'StateHoliday_c']), np.vstack(data_train['Sales'])
dtrain = xgb.DMatrix(
    data=X_train.values,
    feature_names=X_train.columns,
    label=np.log(y_train + 1)
)

X_holdout, y_holdout = data_holdout.drop(columns=['Sales', 'Date', 'Customers', 'StateHoliday_b', 'StateHoliday_c']), np.vstack(data_holdout['Sales'])
dholdout= xgb.DMatrix(
    data=X_holdout.values,
    feature_names=X_holdout.columns,
    label=np.log(y_holdout + 1)
)

X_test = data_test.drop(columns=['Id', 'Date'])
dtest= xgb.DMatrix(
    data=X_test.values,
    feature_names=X_test.columns
)


## preliminary model
### ***CHECK LATER***: have I taken the zeros out for training?
params = {
    'objective': 'reg:squarederror',
    'disable_default_eval_metric': True
}
results = dict()
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=300,
    evals=[(dtrain, 'train'), (dholdout, 'eval')],
    custom_metric=rmspe_xg,
    verbose_eval=True,
    early_stopping_rounds=15,
    evals_result=results
)

# EVALUATE PRELIMINARY MODEL on holdout set
## (NOTE: normally would use a self-generated test dataset for this)

# make predictions for holdout data
y_holdout_pred = model.predict(dholdout)
y_holdout_pred = [math.exp(y_hat) for y_hat in y_holdout_pred]

# evaluate predictions
y_holdout_pred = pd.DataFrame(y_holdout_pred)
y_holdout_pred = y_holdout_pred.astype(int).astype(float)
y_holdout = pd.DataFrame(y_holdout).astype(float)
rmspe(y_holdout_pred.values, y_holdout.values)

# retrieve performance metrics
rmspe_train = pd.DataFrame(results['train'], columns=results['train'].keys())
rmspe_holdout = pd.DataFrame(results['eval'], columns=results['eval'].keys())
epochs = rmspe_holdout.shape[0]

# plot preliminary model's performance on holdout set as rmspe
x_axis = range(0, epochs)
fig, ax = plt.subplots()
ax.plot(x_axis, rmspe_train, label='Train')
ax.plot(x_axis, rmspe_holdout, label='Holdout')
ax.legend()
plt.ylabel('RMSPE')
plt.title('XGBoost RMSPE')
plt.show()


# PREDICTION: create submission file (test data)

# make predictions for test data
y_test_pred = model.predict(dtest)
y_test_pred = [math.exp(y_hat) for y_hat in y_test_pred]
y_test_pred = pd.DataFrame(y_test_pred)
y_test_pred = y_test_pred.astype(int).astype(float)
submission = pd.DataFrame(data_test['Id'])
submission = submission.join(y_test_pred)
submission = submission.rename(columns={0: 'Sales'})
submission.to_csv('submission.csv', index=False)


# MODEL OPTIMIZATION
## - online gibt es gezielte Listen und Tutorials, was man in welcher Reihenfogle optimieren sollte: e.g. https://www.geeksforgeeks.org/xgboost/ or https://machinelearningmastery.com/extreme-gradient-boosting-ensemble-in-python/
## - auch: missing value: statt null mal mean/median versuchen
## - feature selection based on feature importance: https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
## - what else





# ---------------------------------- #



## what coiunts as "highly accurate forecasting": https://plos.figshare.com/articles/dataset/_Criteria_of_MAPE_and_RMSPE_/1560669

# VISUALIZATION

