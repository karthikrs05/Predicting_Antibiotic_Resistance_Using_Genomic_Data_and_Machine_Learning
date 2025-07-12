# code2.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")


def prepare_data_for_analysis(phenotype_column):
    try:
        metadata = pd.read_csv('metadata.csv', index_col=0)
        unitig_data = pd.read_csv(f'{phenotype_column}_gwas_filtered_unitigs.Rtab', sep=" ", index_col=0, low_memory=False)
    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
        return None, None

    metadata = metadata.dropna(subset=[phenotype_column])
    phenotype_data = metadata[phenotype_column]
    unitig_data = unitig_data.transpose()

    common_samples = phenotype_data.index.intersection(unitig_data.index)
    unitig_data = unitig_data.loc[common_samples]
    phenotype_data = phenotype_data.loc[common_samples]

    selector = VarianceThreshold(threshold=0.01)
    X_reduced = selector.fit_transform(unitig_data)
    selected_features = unitig_data.columns[selector.get_support()]
    unitig_data = pd.DataFrame(X_reduced, columns=selected_features)

    return unitig_data, phenotype_data


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = balanced_accuracy_score(y_train, y_train_pred)
    test_acc = balanced_accuracy_score(y_test, y_test_pred)

    cm = confusion_matrix(y_test, y_test_pred)

    # Feature importance if available
    feature_importance = None
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            'Feature (Unitig)': X_train.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        feature_importance = importance_df

    return {
        'model': model_name,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'confusion_matrix': cm,
        'feature_importance': feature_importance
    }


def run_classical_models(phenotype_name):
    X, y = prepare_data_for_analysis(phenotype_name)
    if X is None:
        return []

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = []

    # SVM (scaled)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    svm_model = SVC(kernel='rbf', C=0.5, gamma='scale', class_weight='balanced', random_state=42)
    results.append(train_and_evaluate_model(svm_model,
                                            pd.DataFrame(X_train_scaled, columns=X.columns),
                                            y_train,
                                            pd.DataFrame(X_test_scaled, columns=X.columns),
                                            y_test,
                                            'SVM'))

    # Decision Tree
    dt_model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=42
    )
    results.append(train_and_evaluate_model(dt_model, X_train, y_train, X_test, y_test, 'Decision Tree'))

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    )
    results.append(train_and_evaluate_model(rf_model, X_train, y_train, X_test, y_test, 'Random Forest'))

    return results
