import os
import sys
import dill

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exceptoion import customException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj) 
            
    except Exception as e:
        raise customException(e,sys) 
    
def evaluate_model(X_train,y_train,X_test,y_test,models,params):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param=params[list(models.keys())[i]]
            
            gs = GridSearchCV(model,param,cv=3)
            gs.fit(X_train,y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            
            # model.fit(X_train,y_train)
            
            y_train_preds = model.predict(X_train)
            
            y_test_preds = model.predict(X_test)
            
            train_model_score = r2_score(y_train,y_train_preds)
            
            test_model_score = r2_score(y_test,y_test_preds)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
            
    except Exception as e:
        raise customException(e,sys)
    
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise customException(e,sys)