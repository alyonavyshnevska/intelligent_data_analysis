---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python id="b-0HHZCQp3D2" colab_type="code" outputId="6c749707-6211-478f-af58-202c64c73f56" colab={"base_uri": "https://localhost:8080/", "height": 71}
# Imports
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

#visual
import seaborn as sns 
import matplotlib.pyplot as plt
import missingno as mno
from imblearn.over_sampling import SMOTE

#learning
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import SGDClassifier, RidgeClassifier, LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.gaussian_process.kernels import RBF

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
```

<!-- #region {"id": "RzhzWP6kp3EB", "colab_type": "text"} -->
## Getting to Know the Data
<!-- #endregion -->

```python id="r4IOJuILp3EC" colab_type="code" outputId="f2101711-5aed-4d1a-d9ce-141a6fbcfa5a" colab={"base_uri": "https://localhost:8080/", "height": 34}
# Loading data as pandas dataframe

# Naming the columns with attributes they represent
names = ['account_status', 'duration', 'credit_history',
             'purpose', 'credit_amount', 'savings_account', 'employment_since',
             'installment_rate', 'personal_status_sex', 'guarantors', 
             'residence_since', 'property', 'age', 'other_installment_plans',
             'housing', 'number_of_credits', 'job', 'people_to_maintain',
             'phone', 'foreign_worker', 'creditworthy']

df = pd.read_csv('kredit.dat', header=None, sep="\t", names=names)
df.shape
```

```python id="emkpOKvQp3EK" colab_type="code" outputId="b20b0d31-625f-47e8-cc4f-084756c9b5e4" colab={"base_uri": "https://localhost:8080/", "height": 34}
# Last column is the target variable
df.creditworthy.replace([1,2], [1,0], inplace=True)
label = df['creditworthy']
df = df.drop(columns=['creditworthy'])
df.shape
```

```python id="ppZxcGIgp3EQ" colab_type="code" outputId="95bf18e7-718b-4a64-d847-21dcc1ae8289" colab={"base_uri": "https://localhost:8080/", "height": 258}
# Getting familiar with data

print(f'Number of data points: {df.shape[0]}')
print(f'Number of Attributes: {df.shape[1]}')
df.head(5)
```

```python id="wOa5c0wBp3EU" colab_type="code" outputId="c016038b-8fdf-4c35-d309-1fc9a61ca387" colab={"base_uri": "https://localhost:8080/", "height": 374}
# Explore types of data in dataset
print(df.dtypes)
```

```python id="VrYEBOczp3Ee" colab_type="code" outputId="2bf534f9-cf13-46a5-f745-b0390e9325c7" colab={"base_uri": "https://localhost:8080/", "height": 122}
# As we see above: 2 data types
# Features will be treated separately

def categorical_numerical_split(df):
    categorical = [att for att in df.columns if df[att].dtype == 'object']
    numerical = [att for att in df.columns if np.issubdtype(df[att].dtype, np.number) ]
    return categorical, numerical

categorical, numerical = categorical_numerical_split(df)

print(f'Categorical features:\n{categorical}\n')
print(f'Numerical features:\n{numerical}')
```

```python id="PzM2-Kflp3Ep" colab_type="code" colab={}
# Change a yes/no categorical varibable into numeric one
df.replace('A201', 1, inplace=True)
df.replace('A202', 0, inplace=True)
df.replace('A191', 0, inplace=True)
df.replace('A192', 1, inplace=True)

#Rescale
df.credit_amount = np.log(df.credit_amount)
```

<!-- #region {"id": "fztzqiCJp3Eu", "colab_type": "text"} -->
### 2.2 Handling Missing Values
<!-- #endregion -->

<!-- #region {"id": "6FBBct1bp3Ev", "colab_type": "text"} -->
NaN in Pandas:
1. Nan != NaN (whereas None == None)  
2. When summing data, NaN values will be treated as zero.   
If the data are all NA, the result will be 0.  


<!-- #endregion -->

```python id="JR8UvCG5p3Ew" colab_type="code" colab={}
# For more convenient operations for missing values, replace ? with NaN
# As we see, there are too many missing values to remove these points from the dataset
df.replace("?", np.nan, inplace=True)
```

```python id="TyLPZsjBp3E0" colab_type="code" outputId="9c988d32-453d-4c27-c337-d5d5075b795b" colab={"base_uri": "https://localhost:8080/", "height": 408}
print(f'Number of Missing Values:\n\n{df.isnull().sum()}')
```

```python id="IoZDR0ZKp3E4" colab_type="code" outputId="c0139c9f-b596-46d5-b792-230987b4d046" colab={"base_uri": "https://localhost:8080/", "height": 51}
# Grab missing columns
missing_columns = ['purpose', 'employment_since', 'job', 'foreign_worker']

# create df with features that are fully available
df_X = df.dropna(axis=1, how='any')
print(f'Comparing the data frame with all featrues {df.shape}')
print(f'to the one with with columns where no featrues ar emissing {df_X.shape}')
```

```python id="5nOYxryGp3E6" colab_type="code" outputId="62e188e8-8ddc-47cc-d6b7-3267ac3a4a37" colab={"base_uri": "https://localhost:8080/", "height": 102}
# Percentage of missing values
df[missing_columns].isnull().sum()/len(df)
```

<!-- #region {"id": "PaPBHlblp3E9", "colab_type": "text"} -->
Purpose - 17.3%.   
Employment since - 49.6%  
job - 0.23%.   
foreign worker - 36%. 
<!-- #endregion -->

```python id="d0Kvd8e6p3E_" colab_type="code" outputId="de76b249-8311-4509-e5b2-809a30b03a81" colab={"base_uri": "https://localhost:8080/", "height": 540}
mno.matrix(df, figsize = (10, 6))
```

<!-- #region {"id": "z19bD-wgp3FE", "colab_type": "text"} -->
There are many options we could consider when replacing a missing value, for example:

- A constant value that has meaning within the domain, such as 0, distinct from all other values.  
- A value from another randomly selected record.  
- A mean, median or mode value for the column.  
- A value estimated by another predictive model. 
<!-- #endregion -->

<!-- #region {"id": "_cE5S0VKp3FF", "colab_type": "text"} -->
Prediction model is one of the sophisticated method for handling missing data.   
Here, we create a predictive model to estimate values   
that will substitute the missing data.  
In this case, we divide our data set into two sets:   
One set with no missing values for the variable and another one with missing values.  
First data set become training data set of the model while second data set with missing values   
is test data set and variable with missing values is treated as target variable.  
Next, we create a model to predict target variable based on other attributes   
of the training data set and populate missing values of test data set.


<!-- #endregion -->

```python id="SEoG4Phhp3FI" colab_type="code" outputId="200509c7-2488-4769-8bd3-5501f0a83370" colab={"base_uri": "https://localhost:8080/", "height": 85}
print(df.shape)
# Dropping not an option: too much data lost
print(df.dropna().shape)
#no data points withh all values missing
print(df.dropna(how='all').shape)
# ?? maybe it makes sense to remove data points with many missing things
print(df.dropna(subset=['purpose', 'foreign_worker', 'employment_since'], 
                how='all').shape)
# print(df.purpose.value_counts(dropna=False))
```

<!-- #region {"id": "SXaEDsNrp3FK", "colab_type": "text"} -->
## 2. Data Pre-Processing
<!-- #endregion -->

<!-- #region {"id": "YICKwHobp3FL", "colab_type": "text"} -->
### 2.1 Feature Representation
<!-- #endregion -->

<!-- #region {"id": "xZwA9Xj6p3FN", "colab_type": "text"} -->
#### 2.1.1 Numerical
<!-- #endregion -->

```python id="x0FBkGw2p3FN" colab_type="code" colab={}
# Ranges of values 
def display_range(df, numerical):
    for pos in range(len(numerical)):
        print(f'{numerical[pos]} : {df[numerical[pos]].min()} - \
              {df[numerical[pos]].max()}')
```

```python id="v_OefjRSp3FP" colab_type="code" outputId="92298075-4ed4-45ad-f82a-d1b7d926c839" colab={"base_uri": "https://localhost:8080/", "height": 136}
display_range(df, numerical)
```

```python id="F2Dr_Unyp3FS" colab_type="code" outputId="dff1793c-5ee8-4f4f-ed64-c17383069516" colab={"base_uri": "https://localhost:8080/", "height": 941}
# As we see, the scales are very different, so we will have to normalize data
df.hist(figsize = (10,13))
```

```python id="hSy02RBdp3FW" colab_type="code" colab={}
# Normalize all numerical attributes
def zscore(x):
    #assert isinstance(x,np.ndarray), "x must be a numpy array"
    return (x-np.mean(x)) / np.std(x)
```

```python id="o7sEQSDBp3FY" colab_type="code" outputId="fbd98571-0165-4523-cb30-9dc346e019f0" colab={"base_uri": "https://localhost:8080/", "height": 224}
df = df.apply(lambda x: zscore(x) if x.dtype == 'int64' else x)
df.head()
```

```python id="CgGNMLJwp3Fa" colab_type="code" outputId="20e7b214-e69d-4ff2-88f8-4c549d98ccf1" colab={"base_uri": "https://localhost:8080/", "height": 136}
display_range(df, numerical)
```

```python id="nfQff5GHp3Fc" colab_type="code" outputId="74de51a8-5a33-4804-cbfc-34b2f26b65e9" colab={"base_uri": "https://localhost:8080/", "height": 941}
df.hist(figsize = (10,13))
```

<!-- #region {"id": "K91mr0S5p3Ff", "colab_type": "text"} -->
#### 2.1.2 Categorical

<!-- #endregion -->

```python id="jVOHsRnjp3Fg" colab_type="code" colab={}
# Cols without missing values = X, with missing values become y one by one
```

```python id="PsBYxU69p3Fj" colab_type="code" outputId="78612a96-4ebd-4cb9-8846-15c07fc39b85" colab={"base_uri": "https://localhost:8080/", "height": 255}
# Show how many unique categorical values we have: useful for one-hot encoding
df[categorical].nunique()
```

```python id="QND4PsTkp3Fl" colab_type="code" outputId="3a5f26ef-ee60-498d-c382-3c1886e81dd8" colab={"base_uri": "https://localhost:8080/", "height": 221}
# Feature representation

print("Unique categorical values:")
for att in df.select_dtypes(include=[object]):
    print(att,":", df[att].unique())
    
# We will have to represent features differently 
```

```python id="mNuGy9blp3Fo" colab_type="code" colab={}
def str_to_num(df):
    ''' Encodes nominal features to numeric features 
    return: data frame with all-numeric features, a dict to decode '''
    

    # create a new df with categorical features only, encoded as numbers
    categorical, numerical = categorical_numerical_split(df)
    
    enc = LabelEncoder()
    df_encoded = df[categorical].apply(lambda x: enc.fit_transform(x))

    df_encoded_concat = pd.concat([df_encoded, df[numerical]], axis=1)
    
    # Create a dict to decode numeric values
    d = defaultdict()
    for col_name in df[categorical]:
        # e.g.: account_status_A14 :  3
        for unique_str_val, unique_num_val in zip(
            df[col_name].unique(), df_encoded[col_name].unique()):
                d[col_name + '_' + unique_str_val] = unique_num_val
                
    
    return df_encoded_concat , d

```

```python id="gngY4Gt4p3Fq" colab_type="code" outputId="ee972ea1-c7ae-467d-c1b5-de4c994e96a1" colab={"base_uri": "https://localhost:8080/", "height": 544}
df_X_enc, df_X_enc_dict = str_to_num(df_X)

# Print out new numericaly encoded features
for k,v in df_X_enc_dict.items():
    print(k, ': ', v)

```

```python id="Ua5AxLr3p3Fs" colab_type="code" outputId="351806cf-1567-4522-96a5-122712eb2cab" colab={"base_uri": "https://localhost:8080/", "height": 34}
# Make sure there are still 16 feature
print(df_X_enc.shape)
```

```python id="1nGyEpg5p3Fu" colab_type="code" colab={}
def to_one_hot(df, verbose=False):
    '''Encodes specified columns of a dataframe as one-hot vectors
    
    df: dataframe 
    to_onehot: list of columns to encode
    
    Returns encoded df
    '''
    
    categorical, numerical = categorical_numerical_split(df)
    
    # 1-hot encoding for the categorical variables
    df_only_one_hot = pd.get_dummies(df[categorical])
    
    if verbose == True:
        print(f'new 1-hot-encoded variables:\n {list(df_only_one_hot.columns)}')
    
    # Concatenate numerical and categorical data
    df_X_onehot = pd.concat([df[numerical], df_only_one_hot], axis = 1)
    
    if verbose == True:
        print(f'Shape: {df_X_onehot.shape}')
        df_X_onehot.head()
    
    return df_X_onehot
```

```python id="GWbC-1cEp3Fv" colab_type="code" outputId="7653a3a6-8702-4094-92d8-2a2ddebace6d" colab={"base_uri": "https://localhost:8080/", "height": 88}
# Fatures that shoud be encoded for X_train
df_X_onehot = to_one_hot(df_X, verbose=True)
```

<!-- #region {"id": "0ib5Gmwap3Fy", "colab_type": "text"} -->
### How balanced is the dataset?
<!-- #endregion -->

```python id="34uy-WYEp3F2" colab_type="code" outputId="3ace7211-ce00-4ba5-97d2-cc0f8e35d3e7" colab={"base_uri": "https://localhost:8080/", "height": 333}
# 1 = good, trustworthy
# 0 = bad, Not trustworthy
print(label.value_counts())

label.value_counts().plot(kind='bar', label= 'Target Values')
# There are more than twice as many creditworthy examples
```

<!-- #region {"id": "P4h9ctGMp3F4", "colab_type": "text"} -->
### Balancing the Dataset  
1. Oversampling: It is the process of generating synthetic data that tries   
to randomly generate a sample of the attributes from observations in the minority class.   
The most common technique is called SMOTE (Synthetic Minority Over-sampling Technique).   
In simple terms, it looks at the feature space for the minority class data points   
and considers its k nearest neighbours.

[CBHK2002]	(1, 2) N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer,   
“SMOTE: synthetic minority over-sampling technique,” Journal of artificial intelligence research,   
16, 321-357, 2002.
<!-- #endregion -->

```python id="ry7uh4_-p3F4" colab_type="code" colab={}
def balance_data(df,label):
    sm = SMOTE(sampling_strategy='minority', random_state=7)

    # Fit the model to generate the data.
    cols = df.columns
    df, label = sm.fit_sample(df, label)
    df = pd.DataFrame(df)
    label = pd.Series(label)

    df.columns = cols
    return df,label
```

```python id="-GpvonaKp3F7" colab_type="code" outputId="4431b914-84a1-40ef-cb12-7776cad0c779" colab={"base_uri": "https://localhost:8080/", "height": 333}
label_orig = label
df_X_enc, label_enc = balance_data(df_X_enc,label_orig)
df_X_onehot, label_onehot = balance_data(df_X_onehot, label_orig)

# 1 = good, trustworthy
# 0 = bad, Not trustworthy
print(label_enc.value_counts())

label_enc.value_counts().plot(kind='bar', label= 'Target Values')
# There are more than twice as many creditworthy examples
```

```python id="Yq7eieuAp3F-" colab_type="code" outputId="cfd098b7-1e56-4038-d546-29a37ea31572" colab={"base_uri": "https://localhost:8080/", "height": 350}
print(label_enc.value_counts())

print(df_X_onehot.shape)
label_onehot.value_counts().plot(kind='bar', label= 'Target Values')
```

<!-- #region {"id": "djmyR5Wip3GB", "colab_type": "text"} -->
### Evaluation
<!-- #endregion -->

<!-- #region {"id": "Sd3ojCVpp3GC", "colab_type": "text"} -->
Aim: how many people out of those who thought werr trustworthy, were actually trustworthy

Aim: reduce the number of false positives, to increase precision  
Note: A model that produces no false positives has a precision of 1.0.

![]()
<!-- #endregion -->

```python id="ebnaVbDlp3GD" colab_type="code" colab={}
# Function for evaluation reports
def cross_validate(clf, X_train, y_train, metrics=['precision', 'accuracy']):
    ''' 10-fold Cross Validation on training and validation data 
        Nothing to return '''

    for metric in metrics: 
        scores = cross_val_score(clf, X_train, y_train, cv=10, scoring=metric)
        
        # The mean score and standard deviation of the score estimate
        print("Cross Validation %s: %0.2f (+/- %0.2f)" % (
            metric, scores.mean(), scores.std()))
    
    return 

def train(clf, X_train, y_train):
    '''Fits classifier'''
    print(f'Training a {clf.__class__.__name__}')
    print(f'with a training set size of {len(X_train)}')
    clf.fit(X_train, y_train)
    return clf


def predict(clf, X_test):    
    ''' Predict on unseen test data 
        Return predicted labels '''
    y_pred = clf.predict(X_test)
    return y_pred


def evaluate_test(y_test, y_pred):
    '''Evaluate on Precision and Accuracy'''
    print(f'Test Precision Score: {precision_score(y_test, y_pred)}')
    print(f'Test Accuracy Score: {accuracy_score(y_test, y_pred)}')
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f'Test Confusion Matrix: \n {confusion_matrix(y_test, y_pred)}')
    print(f'Number of False Positives: {fp}')
    
    return 

def train_predict(clf, X_train, y_train, X_test, y_test=None):
    # Train the classifier
    clf = train(clf, X_train, y_train)
    
    # Predict labes
    y_pred = predict(clf, X_test)
    
    return y_pred
```

<!-- #region {"id": "3OB9meOlp3GE", "colab_type": "text"} -->
### Random Forest
<!-- #endregion -->

<!-- #region {"id": "cs798Abcp3GE", "colab_type": "text"} -->
Random forest classifier creates a set of decision trees from randomly selected   
subset of training set. It then aggregates the votes from different decision trees   
to decide the final class of the test object.
Each non-leaf node in this tree is a decision maker.   
Each node carries out a specific test to determine where to go next.
<!-- #endregion -->

```python id="c7J0w4uOp3GF" colab_type="code" outputId="214a7f73-21c8-453e-a5d5-2de493f9e9dc" colab={"base_uri": "https://localhost:8080/", "height": 391}
# Benchmark With One Hot and Labelencoder:

df_Xs = [df_X_enc, df_X_onehot]
labels = [label_enc, label_onehot]
names = ['Numerically Encoded Features: ', 'One-hot Encoded Features']

for df_X_encoded, lab, name in zip(df_Xs, labels, names):
    print('\n', name)
    print(df_X_encoded.shape)
    # Spliting X and y into train and test version
    X_train, X_test, y_train, y_test = train_test_split(
        df_X_encoded, lab, test_size = 0.25, random_state=4)

    clf = RandomForestClassifier(n_estimators=100, random_state=33)
    y_pred = train_predict(clf, X_train, y_train, X_test, y_test)
    evaluate_test(y_test, y_pred)
```

```python id="7sU5xjdPp3GH" colab_type="code" colab={}
# Random Forest: Grid_Search

def grid_search(model, param_grid, X_train, y_train, scoring='precision'): 
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, 
                               scoring=scoring, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f'Best Score: {grid_search.best_score_}')
    print(f'Best Params: {grid_search.best_params_}')
    
    return grid_search.best_estimator_


def randomforestclf_grid_search(X_train, y_train, X_test, y_test):
    
    num_f = X_train.shape[1]
    num_f_less = int(num_f - (num_f/6))
    
    #Seting the Hyper Parameters
    param_grid = {"max_depth": [3, 5, 7, 10, 20, None],
              "n_estimators":[1, 10, 50, 150],
            #defalut “auto”: max_features=sqrt(n_features)
              "max_features": [1, 10, num_f_less, num_f, "auto"], 
              "criterion" : ['gini','entropy']}

    #Creating the classifier
    model = RandomForestClassifier(random_state=33)

    best_estim = grid_search(model, param_grid, X_train, y_train)

    return best_estim
```

<!-- #region {"id": "maYE0MB-p3GN", "colab_type": "text"} -->
**Gini**

* Favors larger partitions. 
*  Uses squared proportion of classes.  
* Perfectly classified, Gini Index would be zero.  
* We want a variable split that has a low Gini Index.

![](data/gini.png)


**Entropy**

* Favors splits with small counts but many unique values.
* Weights probability of class by log(base=2) of the class probability
* A smaller value of Entropy is better.  That makes the difference between the parent node’s entropy larger
* Information Gain is the Entropy of the parent node minus the entropy of the child nodes.

![](data/entropy.png)



<!-- #endregion -->

<!-- #region {"id": "8LpY_7pzp3GN", "colab_type": "text"} -->
### Handling missing Values
<!-- #endregion -->

```python id="oUXsEGgFp3GO" colab_type="code" outputId="ebad1af7-44dc-4cd7-80f1-04c7b91c4d7a" colab={"base_uri": "https://localhost:8080/", "height": 204}
# Three Nominal values and one numeric 1/0 value
df[missing_columns].head()
```

```python id="c5kyLpVYp3GY" colab_type="code" colab={}
def encode_target_var(y_train, method = 'categorical', verbose=False):
    ''' Encode the y label with a specified method''' 
    
    if method == 'categorical':
        enc = LabelEncoder()
        y_train = enc.fit_transform(y_train)
        if verbose: 
            print('\nCategorical encoding of y label. Classes: ', 
                  list(enc.classes_))
    elif method == 'one-hot':
        enc = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
        y_train = enc.fit_transform(y_train)
        print(y_train)
        if verbose: 
            print('One-hot-encoding of y label. Classes: {}', 
                  list(enc.classes_))
    elif method == 'none':
        pass
        
    return y_train


def split_data_missing_vals(df, df_X, target_col, verbose=False):
    ''' Split data into training and test datasets. 
    Test data: missing value in column,
    train data: existing data  
    
    Return 1) x_train with y column in it
        2) x_train withought y_column
        3) y_train
        4) x_test'''
    
    concat_df_target_col = pd.concat([df_X, df[target_col]], axis=1)

    #Select only the rows where the target label is NaN
    X_test = concat_df_target_col[(
        concat_df_target_col[target_col].isna())]
    # original: df with target column in it
    X_train_with_target = concat_df_target_col[(
        concat_df_target_col[target_col].notna())]

    # Separate the y label
    y_train = X_train_with_target[target_col]

    # Remove the y column from the df
    X_test = X_test.drop([target_col], axis=1)
    X_train = X_train_with_target.drop([target_col], axis=1)
    
    if verbose:
        print('Numer of test data points: ', X_test.shape[0])
        print('Numer of train data points: ', X_train_orig.shape[0])
    

    return X_train_with_target, X_train, y_train, X_test


```

```python id="eKsmcHVip3Gd" colab_type="code" colab={}
def compare_classifiers(classifiers, classifier_names, missing_columns, \
                        df_with_na, df_train, verbose=False):

    for col in missing_columns:
        print('\n================================= \n')
        print(f'Linear Classification for featuren "{col}"')

        # Split data. Test data: missing value in column, train data: existing 
        X_train_with_target, X_train, y_train, X_test = split_data_missing_vals(
            df_with_na, df_train, col)

        for clf, name in zip(classifiers, classifier_names): 
            print(f'\nClassifier: {name} \n')
            
            cross_validate(clf, X_train, y_train, metrics=['accuracy', 'f1_micro'])
    return
```

```python id="CXvlNOw9p3Gf" colab_type="code" outputId="17aa035c-a7b0-4686-8309-c9260abcf254" colab={"base_uri": "https://localhost:8080/", "height": 1000}
# Inherently multiclass: RidgeClassifier()
# Multiclass as One-Vs-All: SGDClassifier(), PassiveAggressiveClassifier()
# https://scikit-learn.org/stable/modules/multiclass.html

classifiers = [SGDClassifier(max_iter=1000, tol=1e-3), RidgeClassifier(), 
               LogisticRegression(solver='lbfgs')]
classifier_names = ['SGDClassifier', 'RidgeClassifier', 'LogisticRegression']
missing_columns = ['purpose', 'employment_since', 'job', 'foreign_worker']

compare_classifiers(classifiers, classifier_names, missing_columns, df, df_X_onehot)
```

<!-- #region {"id": "ZmxIhGr4p3Gh", "colab_type": "text"} -->
Only classification of "foreign_worker" performs well enough to use it to fill the missing values.
<!-- #endregion -->

```python id="mWlCnNi6p3Gh" colab_type="code" colab={}
def predict_column(clf, df_with_na, df_train, col): 
    ''' Args: 
    clf: classifier used for prediction
    data: dataframe
    col: name of column to predict
    
    Return 1) predicted column
            2) dataframe with inserted predicted column'''
    
    # Split data. Test data: missing value in column, train data: existing data 
    X_train_with_target, X_train, y_train, X_test = split_data_missing_vals(
            df_with_na, df_train, col)
    
    y_pred = train_predict(clf, X_train, y_train, X_test)

    # Fill only the NAs with predicted values
    X_test[col] = y_pred
    
    # Concat training and test rows
    df_filled = pd.concat([X_train_with_target, X_test], axis=0)
    #sort them back to original order
    df_filled = df_filled.sort_index()
    return df_filled[col]
    
```

```python id="ZfcDe_hPp3Gj" colab_type="code" colab={}
#create DF with filled cols
filled_cols = pd.DataFrame()
```

```python id="wepz1vC8p3Gk" colab_type="code" outputId="7a53a251-0836-4e5a-c60d-f5c3cdf53c8e" colab={"base_uri": "https://localhost:8080/", "height": 51}
# predict a Foreign Worker column
col_foreign_w = predict_column(
    RidgeClassifier(), df, df_X_onehot, "foreign_worker")

#Add Foreign Worker column to the Filled_cols df
filled_cols['foreign_worker_filled'] = col_foreign_w

#Remove it from missing vals
missing_columns.remove('foreign_worker')
```

```python id="rfTjqdrNp3Gn" colab_type="code" outputId="8f3f0c84-8bf0-4015-aa10-ddae473d77ae" colab={"base_uri": "https://localhost:8080/", "height": 153}
#Add Foreign Worker column to df_X
df_X_onehot = (pd.concat([df_X_onehot, filled_cols], axis=1))
df_X_enc = (pd.concat([df_X_enc, filled_cols], axis=1))
print(df_X_onehot.shape)
print(df_X_enc.shape)
df_X_enc.columns
```

```python id="GqiHe5O_p3Gp" colab_type="code" outputId="4baa8487-8d62-4f25-dd17-dc49a29563bd" colab={"base_uri": "https://localhost:8080/", "height": 153}
# Thy the classifier with additional feature
X_train, X_test, y_train, y_test = train_test_split(
    df_X_enc, label_enc, test_size = 0.25, random_state=4)

clf = RandomForestClassifier(n_estimators=100)
y_pred = train_predict(clf, X_train, y_train, X_test, y_test)
evaluate_test(y_test, y_pred)
```

<!-- #region {"id": "0FsTQdD5p3Gs", "colab_type": "text"} -->
Foreign Worker didn't seem to help. But let's look at feature importance. 
<!-- #endregion -->

```python id="qtjP9FX5p3HU" colab_type="code" colab={}
def display_feature_importance(trained_clf):

    # Print the name and gini importance of each feature
    importances = trained_clf.feature_importances_
    print(f'\n Feature Importance, sums up to 1:')
    for feature in zip(X_train.columns, importances):
        print(feature)
        
    # Visualise Feature Importance:
    indices = np.argsort(trained_clf.feature_importances_)
    plt.figure(figsize=(10,8))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='g', align='center')
    plt.yticks(range(len(indices)), [X_train.columns[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    
    return 
```

```python id="TB1cU4jrp3HW" colab_type="code" outputId="6f1d983c-3b56-4e88-a020-abd3831b6a9a" colab={"base_uri": "https://localhost:8080/", "height": 836}
# Add foreign Worker feature to classification
display_feature_importance(clf)
```

<!-- #region {"id": "gwpCkQhCp3HY", "colab_type": "text"} -->
## Feature Engineering


<!-- #endregion -->

```python id="xIhsPDSHp3HZ" colab_type="code" colab={}
def replace_features(df, col_name, replace_to):
    old_enc = list(df[col_name].unique())
    old_enc.remove(np.nan)
    old_enc = sorted(old_enc)

    # change to "emplayed for more than"
    new_enc = replace_to
    df_new_enc = df.replace(old_enc, new_enc)
    return df_new_enc

def compare_replaced_features(df, col_name, exchange_dict,
                         classifiers, classifier_names, df_X_onehot):

    
    for k,v in exchange_dict.items():
        print('\n\n', k)
        df_enc = replace_features(df, col_name, v)
        compare_classifiers(classifiers, classifier_names, [col_name], df_enc, 
                        df_X_onehot)
        
    return


```

<!-- #region {"id": "iMDRMwLhp3Ha", "colab_type": "raw"} -->
Missing: Purpose - 17.3%. 
Employment since - 49.6%. job - 0.23%.  
foreign worker - 36%. 

Attribute 17: (qualitative) 
Job 
A171 : unemployed/ unskilled - non-resident 
A172 : unskilled - resident 
A173 : skilled employee / official 
A174 : management/ self-employed/ 
highly qualified employee/ officer 

An idea: replace by numerical ordered representation. To score this attribute.
<!-- #endregion -->

```python id="a-m1FBjDp3Ha" colab_type="code" outputId="ea056d3c-4e62-4638-c4f4-fb88c5a9904d" colab={"base_uri": "https://localhost:8080/", "height": 1000}
job_replace_to = {
        'Ordinal from 0 to 3' : [0, 1, 2, 3],
        'unemployed/ unskilled = 0, other = 1' : [0, 0, 1, 1],
        'unemployed = 0, else = 1' : [0, 1, 1, 1]
    }

compare_replaced_features(df, 'job', job_replace_to,
                         classifiers, classifier_names, df_X_onehot)
```

<!-- #region {"id": "OfSEmx-Pp3Hc", "colab_type": "text"} -->
LinearSVC(multi_class="crammer_singer" for job:   

Cross Validation accuracy: 0.64 (+/- 0.03)   
Cross Validation f1_micro: 0.64 (+/- 0.03)
<!-- #endregion -->

<!-- #region {"id": "RzYVFLiHp3Hc", "colab_type": "text"} -->
Great! Binarising the feature helps to classify between unemployed / employed. Now the column can be predicted and added to the data.
<!-- #endregion -->

```python id="y7XYNTAWp3Hd" colab_type="code" colab={}
df_job = replace_features(df, "job", [0, 1, 1, 1])
```

```python id="d3LWuXIHp3He" colab_type="code" colab={}
def ridge_grid_search(df_with_na, df_X, target_col):
    
    _, X_train, y_train, X_test = split_data_missing_vals(
        df_with_na, df_X, target_col)

    # prepare a range of alpha values to test
    params = {'solver': ['auto', 'svd', 'lsqr'],
             'alpha': [1,0.1,0.01,0.001,0.0001,0]}
    # create and fit a ridge regression model, testing each alpha
    model = RidgeClassifier()
    
    # optimised by cross-validated grid-search over a parameter grid.
    grid = GridSearchCV(estimator=model, param_grid=params)
    grid.fit(X_train, y_train)
    # summarize the results of the grid search
    print(f'Best Score: {grid.best_score_}')
    print(f'Best alpha: {grid.best_estimator_.alpha}')
    print(f'Best solver: {grid.best_estimator_.solver}')
    
    return grid.best_estimator_
```

```python id="DvihrMs7p3Hf" colab_type="code" outputId="0007cf5d-22f2-4235-9818-b643a8e51156" colab={"base_uri": "https://localhost:8080/", "height": 102}
clf = ridge_grid_search(df_job, df_X_onehot, 'job')

col_job = predict_column(
    clf, df_job, df_X_enc, "job")
filled_cols['job_filled'] = col_job
missing_columns.remove('job')

df_X_onehot = (pd.concat([df_X_onehot, filled_cols['job_filled']], axis=1))
df_X_enc = (pd.concat([df_X_enc, col_job], axis=1))
```

<!-- #region {"id": "BJ5r_oNwp3Hh", "colab_type": "text"} -->
_Random Forest with Foreign Worker and Job_

<!-- #endregion -->

```python id="r-fPDTvzp3Hh" colab_type="code" outputId="4b74952a-517d-4350-e3d9-35364ac8829c" colab={"base_uri": "https://localhost:8080/", "height": 989}
# Add foreign Worker feature to classification
# Thry the classifier with additional feature
X_train, X_test, y_train, y_test = train_test_split(
    df_X_enc, label_enc, test_size = 0.25)

clf = RandomForestClassifier()
y_pred = train_predict(clf, X_train, y_train, X_test, y_test)
evaluate_test(y_test, y_pred)
display_feature_importance(clf)
```

<!-- #region {"id": "Okf8huREp3Hj", "colab_type": "text"} -->
### Employment Since
<!-- #endregion -->

```python id="LeFAExVQp3Hj" colab_type="code" outputId="763716ea-ee94-497d-bc16-7cfdec295c9e" colab={"base_uri": "https://localhost:8080/", "height": 1000}
empl_since_replace_to = {
        'Employed at least n years' : [0, 1, 1, 4, 7],
        'Employed at most n years' : [0, 1, 4, 7, 50],
        'Employed more than a year' : [0, 0, 1, 1, 1],
        'Employed more than 4 years' : [0, 0, 0, 1, 1],
        'Employed more than 7 years' : [0, 0, 0, 1, 1]
    }

compare_replaced_features(df, 'employment_since', empl_since_replace_to,
                         classifiers, classifier_names, df_X_onehot)
```

<!-- #region {"id": "5QQjNtMGp3Hm", "colab_type": "text"} -->
Employed more than a year
Cross Validation accuracy: 0.77 (+/- 0.02)
Cross Validation f1_micro: 0.70 (+/- 0.13)
<!-- #endregion -->

```python id="j5jPn7i4p3Hm" colab_type="code" outputId="1184cf78-2efd-49bd-ef5e-51b391d6d214" colab={"base_uri": "https://localhost:8080/", "height": 102}
df_empl_since = replace_features(df, 'employment_since', [0, 0, 1, 1, 1])

clf = ridge_grid_search(df_empl_since, df_X_onehot, 'employment_since')

col_empl_since = predict_column(
    clf, df_empl_since, df_X_onehot, 'employment_since')
filled_cols['employment_since_filled'] = col_empl_since
missing_columns.remove('employment_since')


df_X_onehot = (pd.concat([df_X_onehot, col_empl_since], axis=1))
df_X_enc = (pd.concat([df_X_enc, col_empl_since], axis=1))
```

```python id="7Q6DNFt4p3Ho" colab_type="code" outputId="39d4e0a4-dd96-417d-9d40-416b67c21a93" colab={"base_uri": "https://localhost:8080/", "height": 34}
df_X_enc.shape
```

```python id="TGeAQzmpp3Hp" colab_type="code" outputId="50bc56fb-a219-4ac1-c341-0b9bd1eed77b" colab={"base_uri": "https://localhost:8080/", "height": 1000}
# Foreign, Job, Empl_since
# Add foreign Worker feature to classification
# Thry the classifier with additional feature
X_train, X_test, y_train, y_test = train_test_split(
    df_X_enc, label_enc, test_size = 0.25)

clf = RandomForestClassifier()
y_pred = train_predict(clf, X_train, y_train, X_test, y_test)
evaluate_test(y_test, y_pred)
display_feature_importance(clf)
```

<!-- #region {"id": "aWOexU9jp3Hq", "colab_type": "text"} -->
*Purpose*
<!-- #endregion -->

```python id="m-ZiwPWUp3Hr" colab_type="code" outputId="b5d67fd0-cab0-4439-f954-472c177937c5" colab={"base_uri": "https://localhost:8080/", "height": 374}
#The are many features available. 
print('Number of non-na values: ', df['purpose'].count())
print(df['purpose'].unique())

col_purpose = df['purpose'].fillna('missing')
col_purpose = encode_target_var(col_purpose, method = 'onehot')
pd.Series(col_purpose).unique()

compare_classifiers(classifiers, classifier_names, missing_columns, df, df_X_onehot)
```

```python id="pARaVZMjp3Hs" colab_type="code" outputId="0c877462-8d37-42b0-9305-7281d9bc3b34" colab={"base_uri": "https://localhost:8080/", "height": 51}
# make sure that the filled df has the right size
print(df_X_enc.shape)
df_X_onehot.shape
```

<!-- #region {"id": "8qN0E7irp3Hu", "colab_type": "text"} -->
### Correlation Matrix between features
<!-- #endregion -->

```python id="8V2nRprKp3Hv" colab_type="code" outputId="fe01fe8f-86e4-4727-aac9-5d8a0f3a45a7" colab={"base_uri": "https://localhost:8080/", "height": 718}
def plot_corr_matrix(df):
    # Sample figsize in inches
    fig, ax = plt.subplots(figsize=(10,10))         
    # Imbalanced DataFrame Correlation
    corr = df.corr()
    sns.heatmap(corr, cmap='YlGnBu', annot_kws={'size':30}, ax=ax)
    ax.set_title("Correlation Matrix", fontsize=14)
    plt.show()
    return

plot_corr_matrix(df_X_enc)
```

```python id="QLSKJVEcp3Hx" colab_type="code" outputId="8ec33f08-e669-4621-d3c3-d3812373eeb7" colab={"base_uri": "https://localhost:8080/", "height": 1000}
# Foreign, Job, Empl_since
# Add foreign Worker feature to classification
# Thry the classifier with additional feature
X_train, X_test, y_train, y_test = train_test_split(
    df_X_enc, label_enc, test_size = 0.25)

clf = randomforestclf_grid_search(X_train, y_train, X_test, y_test)

y_pred = train_predict(clf, X_train, y_train, X_test, y_test)
evaluate_test(y_test, y_pred)
display_feature_importance(clf)

```

```python id="YHB8plJv3P6e" colab_type="code" outputId="2eb2ba9d-2b4c-4bc1-deb1-f782fd74149e" colab={"base_uri": "https://localhost:8080/", "height": 751}
X_train, X_test, y_train, y_test = train_test_split(
    df_X_enc[['account_status', 'savings_account', 'credit_history', 'duration']],
    label_enc, test_size = 0.25)

clf = RandomForestClassifier()

y_pred = train_predict(clf, X_train, y_train, X_test, y_test)
evaluate_test(y_test, y_pred)
display_feature_importance(clf)
```

```python id="4oZ_Y1JHp3Hy" colab_type="code" colab={}
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization by class can be applied by setting `normalize=True`.
    This kind of normalization can be interesting in case of class
    imbalance to have a more visual interpretation of 
    which class is being misclassified.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("\nNormalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
```

```python id="1WciYGQA4-gb" colab_type="code" colab={}
X_train, X_test, y_train, y_test = train_test_split(
    df_X_enc, label_enc, test_size = 0.25)
```

```python id="hQ_1Cd5Qp3H0" colab_type="code" outputId="7e300056-7614-433b-e160-3dd10a9507e5" colab={"base_uri": "https://localhost:8080/", "height": 713}
plot_confusion_matrix(list(y_test), list(y_pred), np.array([0, 1]))
plot_confusion_matrix(list(y_test), list(y_pred), np.array([0, 1]), normalize=True)
```

```python id="stFlbNmnp3H1" colab_type="code" colab={}
def try_other_classifier(clf, X_train, y_train, X_test, y_test):
    y_pred = train_predict(clf, X_train, y_train, X_test, y_test)
    evaluate_test(y_test, y_pred)
    plot_confusion_matrix(list(y_test), list(y_pred), np.array([0, 1]))
    plot_confusion_matrix(list(y_test), list(y_pred), np.array([0, 1]), normalize=True)
```

```python id="e-CY4JdKp3H2" colab_type="code" outputId="aa0d06fe-6c10-481c-b82b-00957d1ec64d" colab={"base_uri": "https://localhost:8080/", "height": 832}
clf_svm = SVC(random_state = 9, kernel='rbf')
try_other_classifier(clf_svm, X_train, y_train, X_test, y_test)
```

```python id="8dXyY8gHp3H4" colab_type="code" outputId="4c718925-8b6f-4233-ddad-0a0ac6672399" colab={"base_uri": "https://localhost:8080/", "height": 832}
clf_svm = SVC(random_state = 9, kernel='linear')
try_other_classifier(clf_svm, X_train, y_train, X_test, y_test)
```

```python id="YpZAhqk1p3H7" colab_type="code" outputId="cd4926d1-9a93-4bba-c043-58e2c19f96dc" colab={"base_uri": "https://localhost:8080/", "height": 832}
clf_log_reg = LogisticRegression()
try_other_classifier(clf_log_reg, X_train, y_train, X_test, y_test)
```

```python id="jG_q7dwAp3H8" colab_type="code" outputId="f058d09e-0e48-4d37-b660-07f2fa9d04b4" colab={"base_uri": "https://localhost:8080/", "height": 832}
clf_extra = ExtraTreesClassifier()
try_other_classifier(clf_extra, X_train, y_train, X_test, y_test)
```
