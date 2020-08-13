# computation
import numpy as np

# preprocessing data
from sklearn.preprocessing import StandardScaler

#model_selection
from sklearn.model_selection import cross_val_score, GridSearchCV

# models
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

import xgboost as xgb



class Model:
    """
    Represent a model
    attributes:
    ----------
        model: the model's class instantiate
        name: the model's name
        params: the model's params to tune
        score: the score perform by the model
    """
    
    def __init__(self, model, name, params=None):
        self.model = model
        self.name = name
        self.params = params
        self._score = 0
        
    def _get_score(self):
        return self._score
    
    def _set_score(self, score):
        self._score = score
        
    score = property(_get_score, _set_score)
    
    
class Tester:
    """
    Represent a tester who test a list of models with some data
    attributes:
    ----------
        models: a list of Model
    """
    
    def __init__(self, models, X, y):
        self.models = models
        self.X = X
        self.y = y
        
    def test(self):
        run(self.models, self.X, self.y)
        
    def best_model(self):
        return max(self.models, key=(lambda model: model.score))
        
        
def build_test(model, X, y):
    """
    Build the model and evaluate it score via a cross-validation using X and y,
    if params is specified a GridSearchCV is make.
    parameters:
    ----------
        model: the Model to build and evaluate
        X: the feature matrix
        y: the target vector
    """
    print(model.name)
    
    if model.params:
        
        model.model = GridSearchCV(model.model, model.params, cv=10, scoring="roc_auc")
        model.model.fit(X,y)
        model.score = model.model.best_score_
        
        print("  Best parameter: {}".format(model.model.best_params_))
        print("  Best roc_auc_score: {} ({}%)".format(model.score, round(model.score*100, 3)))
              
    else:
        model.score = cross_val_score(model.model, X, y, cv=10, scoring="roc_auc").mean()
        print("  Roc_auc_score: {} ({}%)".format(model.score, round(model.score*100, 3)))
    
    print("\n")
        
        
def run(models, X, y):
    """
    Run all the models with X, y
    parameters:
    ----------
        models: list of Model
        X: feature matrix
        y: target vector
    """
    for model in models:
        build_test(model, X, y)
    
    print("All tests are done")
    

def get_X(data, features):
    """
    return the features matrix
    Parameters:
    -----------
        data: data from which we retrieve the features matrix
        features: the features we want into the features matrix
    """
    
    # features matrix
    X = data.loc[:, features]

    # data scale
    return StandardScaler().fit_transform(X)
        
def get_models():
    """
    Return the models for our prediction
    """
    
    model1 = Model(LogisticRegression(), "Logistic Regression")
    model2 = Model(KNeighborsClassifier(), "KNN", {"n_neighbors": np.arange(1,12)})
    model3 = Model(SVC(), "SVC", {"degree": np.arange(1,10)})
    model4 = Model(AdaBoostClassifier(), "AdaBoost", {"n_estimators": np.arange(1,30)})
    model5 = Model(xgb.XGBClassifier(objective="binary:logistic", missing=None, seed=42), "XgBoost")

    return [model1, model2, model3, model4, model5]
        
        
        
        
        
        
        
        
        
        
        
        
        
        