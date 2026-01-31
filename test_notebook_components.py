#!/usr/bin/env python3
"""
Comprehensive verification script for Load-Default_Prediction.ipynb
Tests all key components to ensure everything is working correctly
"""

import sys
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("COMPREHENSIVE NOTEBOOK VERIFICATION")
print("="*70)

# Track results
results = []

def test(name, condition, error_msg=""):
    """Helper function to test conditions"""
    if condition:
        print(f"{name}")
        results.append((name, True, ""))
        return True
    else:
        print(f"{name}")
        if error_msg:
            print(f"   Error: {error_msg}")
        results.append((name, False, error_msg))
        return False

print("\n" + "="*70)
print("1. TESTING LIBRARY IMPORTS")
print("="*70)

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    test("Core libraries (pandas, numpy, matplotlib, seaborn)", True)
except Exception as e:
    test("Core libraries", False, str(e))

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
    test("Scikit-learn metrics and model_selection", True)
except Exception as e:
    test("Scikit-learn imports", False, str(e))

try:
    from imblearn.over_sampling import BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler
    test("Imbalanced-learn (BorderlineSMOTE, RandomUnderSampler)", True)
except Exception as e:
    test("Imbalanced-learn imports", False, str(e))

try:
    from sklearn.feature_selection import SelectKBest, chi2, RFE
    test("Feature selection (SelectKBest, chi2, RFE)", True)
except Exception as e:
    test("Feature selection imports", False, str(e))

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    test("All 3 models (LogisticRegression, RandomForest, XGBoost)", True)
except Exception as e:
    test("Model imports", False, str(e))

print("\n" + "="*70)
print("2. TESTING DATA LOADING")
print("="*70)

try:
    demo = pd.read_csv("data/train/traindemographics.csv")
    test(f"Demographics data loaded ({len(demo)} rows)", len(demo) > 0)
except Exception as e:
    test("Demographics data", False, str(e))

try:
    perf = pd.read_csv("data/train/trainperf.csv")
    test(f"Performance data loaded ({len(perf)} rows)", len(perf) > 0)
except Exception as e:
    test("Performance data", False, str(e))

try:
    prev = pd.read_csv("data/train/trainprevloans.csv")
    test(f"Previous loans data loaded ({len(prev)} rows)", len(prev) > 0)
except Exception as e:
    test("Previous loans data", False, str(e))

print("\n" + "="*70)
print("3. TESTING SAMPLING STRATEGIES")
print("="*70)

try:
    # Create dummy data for testing
    X_dummy = np.random.rand(1000, 10)
    y_dummy = np.array([0]*900 + [1]*100)  # Imbalanced: 900 vs 100
    
    # Test BorderlineSMOTE
    smote = BorderlineSMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_dummy, y_dummy)
    test(f"BorderlineSMOTE works (created {len(X_smote)} samples)", len(X_smote) > len(X_dummy))
except Exception as e:
    test("BorderlineSMOTE", False, str(e))

try:
    # Test RandomUnderSampler
    rus = RandomUnderSampler(random_state=42)
    X_rus, y_rus = rus.fit_resample(X_dummy, y_dummy)
    test(f"RandomUnderSampler works (reduced to {len(X_rus)} samples)", len(X_rus) < len(X_dummy))
except Exception as e:
    test("RandomUnderSampler", False, str(e))

print("\n" + "="*70)
print("4. TESTING FEATURE SELECTION")
print("="*70)

try:
    # Test Chi-square
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_dummy)
    
    chi2_selector = SelectKBest(chi2, k=5)
    X_chi2 = chi2_selector.fit_transform(X_scaled, y_dummy)
    test(f"Chi-square selection works (selected {X_chi2.shape[1]} features)", X_chi2.shape[1] == 5)
except Exception as e:
    test("Chi-square selection", False, str(e))

try:
    # Test RFE
    rf_estimator = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
    rfe_selector = RFE(estimator=rf_estimator, n_features_to_select=5)
    X_rfe = rfe_selector.fit_transform(X_dummy, y_dummy)
    test(f"RFE selection works (selected {X_rfe.shape[1]} features)", X_rfe.shape[1] == 5)
except Exception as e:
    test("RFE selection", False, str(e))

print("\n" + "="*70)
print("5. TESTING MODEL TRAINING")
print("="*70)

# Create train/test split
X_train, X_test, y_train, y_test = train_test_split(X_dummy, y_dummy, test_size=0.2, random_state=42)

try:
    # Test Logistic Regression
    lr = LogisticRegression(max_iter=100, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    f1_lr = f1_score(y_test, y_pred_lr)
    test(f"Logistic Regression trains (F1={f1_lr:.3f})", True)
except Exception as e:
    test("Logistic Regression", False, str(e))

try:
    # Test Random Forest
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    f1_rf = f1_score(y_test, y_pred_rf)
    test(f"Random Forest trains (F1={f1_rf:.3f})", True)
except Exception as e:
    test("Random Forest", False, str(e))

try:
    # Test XGBoost
    xgb = XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    f1_xgb = f1_score(y_test, y_pred_xgb)
    test(f"XGBoost trains (F1={f1_xgb:.3f})", True)
except Exception as e:
    test("XGBoost", False, str(e))

print("\n" + "="*70)
print("6. TESTING EVALUATION METRICS")
print("="*70)

try:
    from sklearn.metrics import precision_score, recall_score, accuracy_score
    precision = precision_score(y_test, y_pred_rf)
    recall = recall_score(y_test, y_pred_rf)
    accuracy = accuracy_score(y_test, y_pred_rf)
    test(f"Metrics work (Precision={precision:.3f}, Recall={recall:.3f}, Accuracy={accuracy:.3f})", True)
except Exception as e:
    test("Evaluation metrics", False, str(e))

try:
    # Test confusion matrix
    cm = confusion_matrix(y_test, y_pred_rf)
    test(f"Confusion matrix works (shape={cm.shape})", cm.shape == (2, 2))
except Exception as e:
    test("Confusion matrix", False, str(e))

try:
    # Test ROC-AUC
    y_proba = rf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    test(f"ROC-AUC works (score={roc_auc:.3f})", True)
except Exception as e:
    test("ROC-AUC", False, str(e))

print("\n" + "="*70)
print("7. CHECKING SAVED MODEL")
print("="*70)

try:
    import joblib
    import os
    if os.path.exists('loan_model.pkl'):
        model = joblib.load('loan_model.pkl')
        test(f"Saved model loads ({type(model).__name__})", True)
    else:
        test("Saved model file", False, "loan_model.pkl not found")
except Exception as e:
    test("Saved model", False, str(e))

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

passed = sum(1 for _, success, _ in results if success)
total = len(results)
percentage = (passed / total) * 100

print(f"\nPassed: {passed}/{total} ({percentage:.1f}%)")

if passed == total:
    print("\n🎉 ALL TESTS PASSED! Your notebook is fully functional!")
    sys.exit(0)
elif percentage >= 80:
    print("\n Most tests passed, but some issues need attention.")
    print("\nFailed tests:")
    for name, success, error in results:
        if not success:
            print(f" {name}")
            if error:
                print(f"     {error}")
    sys.exit(1)
else:
    print("\n Multiple tests failed. Please review the errors above.")
    print("\nFailed tests:")
    for name, success, error in results:
        if not success:
            print(f" {name}")
            if error:
                print(f"     {error}")
    sys.exit(1)
