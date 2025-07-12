# xgboost.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier, plot_importance

warnings.filterwarnings("ignore")


def run_xgboost_model(phenotype):
    try:
        metadata = pd.read_csv('metadata.csv', index_col=0)
        unitig_data = pd.read_csv(f'{phenotype}_gwas_filtered_unitigs.Rtab', sep=' ', index_col=0, low_memory=False)
    except FileNotFoundError as e:
        print(f"Missing file: {e.filename}")
        return {
            'model': 'XGBoost',
            'train_accuracy': None,
            'test_accuracy': None,
            'confusion_matrix': None,
            'feature_importance': None
        }

    metadata = metadata.dropna(subset=[phenotype])
    phenotype_data = metadata[phenotype]
    unitig_data = unitig_data.transpose()

    common_samples = phenotype_data.index.intersection(unitig_data.index)
    X = unitig_data.loc[common_samples]
    y = phenotype_data.loc[common_samples]

    print(f"{phenotype.upper()} - Samples: {len(common_samples)}")

    # 1. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 2. Param grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    xgb = XGBClassifier(objective='binary:logistic',
                        eval_metric='logloss',
                        use_label_encoder=False,
                        n_jobs=-1,
                        random_state=42)

    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='balanced_accuracy',
        cv=cv,
        n_jobs=-1,
        verbose=0
    )

    # 3. Train
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_score = grid_search.best_score_

    # 4. Predict
    y_pred = best_model.predict(X_test)
    train_pred = best_model.predict(X_train)

    test_acc = balanced_accuracy_score(y_test, y_pred)
    train_acc = balanced_accuracy_score(y_train, train_pred)
    cm = confusion_matrix(y_test, y_pred)

    # 5. Feature importance
    importance_df = pd.DataFrame({
        'Feature (Unitig)': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # 6. Return dictionary
    return {
        'model': 'XGBoost',
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'confusion_matrix': cm,
        'feature_importance': importance_df,
        'cv_score': cv_score,
        'best_params': best_params
    }
