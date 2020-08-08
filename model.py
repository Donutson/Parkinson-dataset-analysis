#model_selection
from sklearn.model_selection import cross_val_score, GridSearchCV



def build_test(model, name, X, y, params=None):
    """
    Build the model and evaluate it score via a cross-validation,
    if params is specified a GridSearchCV is make.
    parameters:
    ----------
        model: the model to build and evaluate
        name: the model's name
        X: the feature matrix
        y: the target vector
        params: a dict of params for the GridSearchCV
    """
    print(name+" :")
    
    if params:
        
        gs_model = GridSearchCV(model, params, cv=10, scoring="roc_auc")
        gs_model.fit(X,y)
        
        print("  Best parameter: {}".format(gs_model.best_params_))
        print("  Best roc_auc_score: {} ({}%)".format(gs_model.best_score_, round(gs_model.best_score_*100, 3)))
              
    else:
        cv_scores = cross_val_score(model, X, y, cv=10, scoring="roc_auc")
        print("  Roc_auc_score: {} ({}%)".format(cv_scores.mean(), round(cv_scores.mean()*100, 3)))
    
    print("\n")
        
        
def run(models, X, y):
  """
  Run all the models with X, y
  parameters:
  ----------
      models: list of tuples, each tuple contains the model's class instantiate, it name and an optional dict of params for GridSearchCV 
      X: feature matrix
      y: target vector
  """
              
  for model in models:
    if len(model) == 3:
        
        build_test(*model[:2], X, y, model[2])
        print("All test are done")
        
    elif len(model) == 2:
        
        build_test(*model, X, y)
        print("All test are done")
        
    else:
      raise TypeError("arg 'models' is bad define")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        