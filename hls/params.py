# -*- coding: utf-8 -*-


def get_defparams(model_name):
    
    if model_name == 'xgb':
        return {'learning_rate': 0.1,
                'n_estimators': 60,
                'max_depth': 3,
                'min_child_weight': 1,
                'subsample': 1,
                'colsample_bytree': 1,
                'gamma':0 }
        
    elif model_name == 'lasso':
        return {'alpha': 20.0}