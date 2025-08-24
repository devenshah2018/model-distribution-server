from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR

MODEL_REGISTRY = {
    'rfr': {
        'constructor': lambda: RandomForestRegressor(n_estimators=50),
        'type': 'rf_regressor',
        'category': 'regression',
        'display_name': 'Random Forest Regressor'
    },
    'svm': {
        'constructor': lambda: SVR(),
        'type': 'svm_regressor',
        'category': 'regression',
        'display_name': 'Support Vector Machine Regressor'
    },
    'rfc': {
        'constructor': lambda: RandomForestClassifier(n_estimators=50),
        'type': 'rf_classifier',
        'category': 'classification',
        'display_name': 'Random Forest Classifier'
    }
}

__all__ = ['MODEL_REGISTRY']