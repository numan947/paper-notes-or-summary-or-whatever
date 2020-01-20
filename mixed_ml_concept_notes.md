# Miscellaneous Machine Learning Things

## Chi2 Test

ML problem: how to determine whether input features are relevant for the outcome to be predicted ==> Feature Selection problem

*Classification problem with Categorical input* -> determine whether output variable is dependent or independent of the input variable -> if independent, input variable is not a good predictor

*Pearson Chi2 Test* -- test for independence between categorical variables --> quantifies the independece of pairs of categorical variables

**Contingency Table**: Table containing summaries of two different categorical variables. One variable's values are in columns while the other's values occupy the rows. Each cell contains the count of observation that has those 2 values for the corresponding variables.

**Chi2 Test**: This is a statistical hypothesis test that assumes that the observed frequencies for a categorical variable match the expected frequencies for the variable -- it has a chi-squared distribution.

Calculate expected frequencies for the groups --> determine whether the division of the groups matches the expected frequencies. The variables are considered independent if observed and expected frequencies are similar.

if Statistic >= Critical value --> NULL hypothesis rejected
else, failed to reject NULL hypothesis

Degrees of freedom: (rows-1)*(cols-1)

In terms of p-value and a chosen significance level:

if, p-value<=alpha --> NULL hypothesis rejected
else, failed to reject NULL hypothesis

**Effective tests required atleast 5 observations in each cell of the contingency table**

Can be used using *SciPy--stats--chi2_contingency(table)*, returns chi2, p, dof, expected


## Going through [Solution to Categorical Feature Encoding Challenge - I](https://www.kaggle.com/adaubas/2nd-place-solution-categorical-fe-callenge#First-logistic-ridge-with-some-one-hot-encoders-only)

- Often simpler model results in better accuracy
- If the relationship between each predictor and target is linear, it's better to use simple linear models like Linear/Logistic Regression
- Acquired some cool plotting technique for Categorical variable: see my utils
- Acquired cool technique to show correlation-like measure for Categorical variable: see my utils
  
- Learned how to create pipelines more effectively
```python
ohe1 = (
    OneHotEncoder(dtype = 'uint8', handle_unknown="ignore"),
    [f for f in train.columns if not f in ["target"]]
)

pipe = make_pipeline(
    make_column_transformer(ohe1), 
    LogisticRegression(**lr_params)
    )
```

- Learned how to use cross validation more effectively
```python
# To avoid target leakage
folds1 = StratifiedKFold(n_splits=__n_folds, shuffle=True, random_state=__seed)
folds2 = StratifiedKFold(n_splits=__n_folds, shuffle=True, random_state=__seed+2)
folds3 = StratifiedKFold(n_splits=__n_folds, shuffle=True, random_state=__seed+4)


def cross_val_print(pipe, X, y, cv, scoring="roc_auc",best_score=0):
    ''' 
    Cross validate score and print result and print previous result
    And show the score and the previous best score.
    '''

    scores = cross_validate(pipe, X, y, cv = cv, scoring = scoring, return_train_score = True)
    cv_score = scores["test_score"].mean()
    
    if cv == folds1:
        precision = 1
    elif cv == folds2: 
        precision = 2
    else: 
        precision = 3

    print("CV{} score on valid : {:.7f}  - Previous best valid score : {:.7f} - Train mean score : {:6f}".\
          format(precision, cv_score, best_score, scores["train_score"].mean()))
    
    if cv_score > best_score:
        best_score = cv_score

    return cv_score, best_score



cross_val_print(pipe, train, train["target"], cv=folds1)

```



- Learned how to transform Categorical variable to Ordinal variable effectively

```python

def transf_ordinal_features(serie):
    
    transform_dict= {
        "ord_1":{'Novice':0, 'Contributor':1, 'Expert':2, 'Master':3, 'Grandmaster':4},
        "ord_2":{'Freezing':0, 'Cold':1, 'Warm':2, 'Hot':3, 'Boiling Hot':4, 'Lava Hot':5},
        "nom_0":{"Blue":1, "Green":2, "Red":3},
        "nom_1":{"Circle":1, "Trapezoid":2, "Star":3, "Polygon":4, "Square":5, "Triangle":6},
        "nom_2":{"Dog":1, "Lion":2, "Snake":3, "Axolotl":4, "Cat":5, "Hamster":6},
        "nom_3":{"Finland":1, "Russia":2, "China":3, "Costa Rica":4, "Canada":5, "India":6},
        "nom_4":{"Bassoon":1, "Piano":2, "Oboe":3, "Theremin":4},
        "bin_0_bin_3_bis":{"T0":0, "F1":2, "F0":1}
        }

    
    if serie.name == "ord_0":
        new_serie = serie - 1 # make the series 0-based
    elif serie.name == "ord_5":
        lm = serie.unique()
        new_serie = serie.map({l:i for i, l in enumerate(list(np.sort(lm))})
    elif serie.name in ["ord_3", "ord_4"]:
        new_serie = serie.str.lower().map({l:i for i, l in enumerate(list(ascii_lowercase))})
    else:
        new_serie = serie.map(dtransf[serie.name])
        
    return new_serie
```


- Learned how to Bin high cardianlity categorical variable using target mean as reference

```python

class MyBinsEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, nbins=100, nmin=20):
        self.nbins = nbins
        self.nmin = nmin
        
    def fit(self, x, y=None):
        
        temp = pd.concat([x, y], axis=1)
        
        # Compute target mean for each value
        averages = temp.groupby(by=x.name)[y.name].mean()
        means = dict(zip(averages.index.values, averages.values))
        
        # binning in self.nbins bins
        bins = np.linspace(averages.min(), averages.max(), self.nbins)
        self.map_ = dict(zip(averages.index.values, np.digitize(averages.values, bins=bins)))

        # But if there are more than self.nmin observations in a original value, keep the original value
        # instead of bins.
        count = temp.groupby(by=x.name)[y.name].count()
        nobs = dict(zip(averages.index.values, count))
        
        for key, value in nobs.items():
            if value > self.nmin:
                self.map_[key] = key
        
        return self
    
    def transform(self, x, y=None):
        
        temp = x.map(self.map_)
        # Especially for nom_7, nom_8 and nom_9
        temp.fillna(random.choice(list(self.map_.values())), inplace=True)
        temp = temp.astype(str)
        
        return temp
```


- Learned how to create custom feature engineering class for different featuresSS

```python
class MyFeaturesEngineering(BaseEstimator, TransformerMixin):
    
    def __init__(self, list_ordinal_features =[], feat_to_bins_encode = {}):
        
        self.list_ordinal_features = list_ordinal_features
        
        self.feat_to_bins_encode = feat_to_bins_encode
        self.BinsEncoder={}
        

    def fit(self, x, y=None):
        
        # bins encoders
        for feat, value in self.feat_to_bins_encode.items():
            self.BinsEncoder[feat] = MyBinsEncoder(nbins=value[0], nmin=value[1])
            self.BinsEncoder[feat].fit(x[feat], y)

        return self
    

    def transform(self, x, y=None):
        
        df = x.copy()
        
        for v in self.feat_to_bins_encode.keys():
            df[v] = self.BinsEncoder[v].transform(df[v])
            
        # Ordinal features
        for v in self.list_ordinal_features:
            df[v] = transf_ordinal_features(df[v])
        
        return df



    ## APPLY THE FEATURE ENGINEERING
    ohe1 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', handle_unknown="ignore")
            , ohe1_feats)

    MyFeE = MyFeaturesEngineering(list_ordinal_features = ordinal_features
                                 , feat_to_bins_encode={feat:nbins})

    StdScalE = (StandardScaler(copy=False), ordinal_features)
    
    pipe = make_pipeline(MyFeE, make_column_transformer(ohe1, StdScalE), LogisticRegression(**lr_params))
    _, best_score1 = cross_val_and_print(pipe, best_score=best_score1, comment1=feat, comment2=nbins)
```