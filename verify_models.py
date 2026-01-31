#!/usr/bin/env python3
"""
Quick verification script to test all models work correctly
"""

print("=" * 60)
print("MODEL VERIFICATION TEST")
print("=" * 60)

# Test 1: XGBoost
print("\n1. Testing XGBoost...")
try:
    from xgboost import XGBClassifier
    print("   ✅ XGBoost imported successfully!")
except Exception as e:
    print(f"   ❌ XGBoost import failed: {e}")

# Test 2: Random Forest
print("\n2. Testing Random Forest...")
try:
    from sklearn.ensemble import RandomForestClassifier
    print("   ✅ Random Forest imported successfully!")
except Exception as e:
    print(f"   ❌ Random Forest import failed: {e}")

# Test 3: Logistic Regression
print("\n3. Testing Logistic Regression...")
try:
    from sklearn.linear_model import LogisticRegression
    print("   ✅ Logistic Regression imported successfully!")
except Exception as e:
    print(f"   ❌ Logistic Regression import failed: {e}")

# Test 4: BorderlineSMOTE
print("\n4. Testing BorderlineSMOTE...")
try:
    from imblearn.over_sampling import BorderlineSMOTE
    print("   ✅ BorderlineSMOTE imported successfully!")
except Exception as e:
    print(f"   ❌ BorderlineSMOTE import failed: {e}")

# Test 5: Saved model
print("\n5. Testing saved model (loan_model.pkl)...")
try:
    import joblib
    model = joblib.load('loan_model.pkl')
    print(f"   ✅ Model loaded: {type(model).__name__}")
    print(f"   ✅ Model params: n_estimators={model.n_estimators}, random_state={model.random_state}")
except Exception as e:
    print(f"   ❌ Model loading failed: {e}")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE!")
print("=" * 60)
