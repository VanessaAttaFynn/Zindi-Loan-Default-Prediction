#!/usr/bin/env python3
"""
Script to enhance the Load-Default_Prediction.ipynb notebook with:
1. Random Undersampling (second sampling strategy)
2. Chi-square feature selection
3. RFE feature selection
4. Comprehensive model comparison framework
"""

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell
import sys

def find_cell_index(nb, search_text, cell_type='code'):
    """Find the index of a cell containing specific text."""
    for idx, cell in enumerate(nb.cells):
        if cell['cell_type'] == cell_type and search_text in cell.get('source', ''):
            return idx
    return -1

def add_random_undersampling(nb):
    """Add Random Undersampling implementation."""
    print("Adding Random Undersampling...")
    
    # Find the cell after BorderlineSMOTE or imblearn import
    smote_idx = find_cell_index(nb, 'BorderlineSMOTE')
    
    if smote_idx == -1:
        smote_idx = find_cell_index(nb, 'from imblearn')
    
    if smote_idx == -1:
        print("❌ Could not find appropriate location for Random Undersampling")
        return False
    
    # Create cells
    cells_to_add = [
        new_markdown_cell("""### 4.2.2 Random Undersampling

Random Undersampling is an alternative class imbalance handling strategy that reduces the majority class by randomly removing samples. This is the opposite approach to oversampling.

**Advantages:**
- Reduces training time (smaller dataset)
- No risk of overfitting from synthetic samples
- Simple and fast to implement

**Disadvantages:**
- May lose important information from majority class
- Can lead to underfitting if too aggressive
"""),
        
        new_code_cell("""# Import Random Undersampling
from imblearn.under_sampling import RandomUnderSampler

# Apply Random Undersampling
print("\\n" + "="*70)
print("RANDOM UNDERSAMPLING")
print("="*70)

rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

print(f"\\nOriginal dataset shape: {X_train.shape}")
print(f"Undersampled dataset shape: {X_train_rus.shape}")
print(f"\\nClass distribution after Random Undersampling:")
print(pd.Series(y_train_rus).value_counts())
print(f"\\nClass balance ratio: {pd.Series(y_train_rus).value_counts()[1] / pd.Series(y_train_rus).value_counts()[0]:.2f}")
"""),
        
        new_code_cell("""# Visualize comparison of sampling strategies
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Original distribution
pd.Series(y_train).value_counts().plot(kind='bar', ax=axes[0], color=['#e74c3c', '#2ecc71'])
axes[0].set_title('Original Class Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Class (0=Bad, 1=Good)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_xticklabels(['Bad (0)', 'Good (1)'], rotation=0)

# After BorderlineSMOTE
if 'y_train_smote' in locals():
    pd.Series(y_train_smote).value_counts().plot(kind='bar', ax=axes[1], color=['#e74c3c', '#2ecc71'])
    axes[1].set_title('After BorderlineSMOTE (Oversampling)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Class (0=Bad, 1=Good)', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_xticklabels(['Bad (0)', 'Good (1)'], rotation=0)

# After Random Undersampling
pd.Series(y_train_rus).value_counts().plot(kind='bar', ax=axes[2], color=['#e74c3c', '#2ecc71'])
axes[2].set_title('After Random Undersampling', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Class (0=Bad, 1=Good)', fontsize=12)
axes[2].set_ylabel('Count', fontsize=12)
axes[2].set_xticklabels(['Bad (0)', 'Good (1)'], rotation=0)

plt.tight_layout()
plt.show()

print("\\n📊 Sampling Strategy Comparison:")
print(f"Original: {len(y_train)} samples")
if 'y_train_smote' in locals():
    print(f"BorderlineSMOTE: {len(y_train_smote)} samples (+{len(y_train_smote) - len(y_train)})")
print(f"Random Undersampling: {len(y_train_rus)} samples ({len(y_train_rus) - len(y_train)})")
""")
    ]
    
    # Insert cells after SMOTE
    insert_idx = smote_idx + 1
    for i, cell in enumerate(cells_to_add):
        nb.cells.insert(insert_idx + i, cell)
    
    print("✅ Random Undersampling added successfully")
    return True

def add_feature_selection(nb):
    """Add Chi-square and RFE feature selection methods."""
    print("Adding Feature Selection methods...")
    
    # Find appropriate location (look for model training)
    model_idx = find_cell_index(nb, 'LogisticRegression')
    if model_idx == -1:
        model_idx = find_cell_index(nb, 'RandomForest')
    
    if model_idx == -1:
        print("❌ Could not find appropriate location for feature selection")
        return False
    
    # Create cells
    cells_to_add = [
        new_markdown_cell("""## 5. Feature Selection

Feature selection is crucial for improving model performance and reducing overfitting. We'll implement two methods:

1. **Chi-square Test (Filter Method)**: Statistical test for categorical target
2. **Recursive Feature Elimination (RFE)**: Wrapper method using model feedback

**Benefits:**
- Reduces dimensionality
- Improves model interpretability
- Reduces overfitting
- Decreases training time
"""),
        
        new_code_cell("""### 5.1 Chi-square Feature Selection

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

print("="*70)
print("CHI-SQUARE FEATURE SELECTION")
print("="*70)

# Ensure all features are non-negative for chi2 test
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Select top 20 features using chi-square
k_features = min(20, X_train.shape[1])  # Select top 20 or all if less
chi2_selector = SelectKBest(chi2, k=k_features)
chi2_selector.fit(X_train_scaled, y_train)

# Get selected features
chi2_support = chi2_selector.get_support()
chi2_features = X_train.columns[chi2_support].tolist()

print(f"\\nSelected {len(chi2_features)} features using Chi-square test:")
print(chi2_features[:10], "..." if len(chi2_features) > 10 else "")

# Get feature scores
chi2_scores = pd.DataFrame({
    'Feature': X_train.columns,
    'Chi2_Score': chi2_selector.scores_
}).sort_values('Chi2_Score', ascending=False)

print(f"\\nTop 10 features by Chi-square score:")
print(chi2_scores.head(10))
"""),
        
        new_code_cell("""### 5.2 Recursive Feature Elimination (RFE)

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

print("\\n" + "="*70)
print("RECURSIVE FEATURE ELIMINATION (RFE)")
print("="*70)

# Use Random Forest as the estimator for RFE
rf_estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

# Select top 20 features using RFE
rfe_selector = RFE(estimator=rf_estimator, n_features_to_select=k_features, step=1)
rfe_selector.fit(X_train, y_train)

# Get selected features
rfe_support = rfe_selector.get_support()
rfe_features = X_train.columns[rfe_support].tolist()

print(f"\\nSelected {len(rfe_features)} features using RFE:")
print(rfe_features[:10], "..." if len(rfe_features) > 10 else "")

# Get feature rankings
rfe_rankings = pd.DataFrame({
    'Feature': X_train.columns,
    'RFE_Ranking': rfe_selector.ranking_
}).sort_values('RFE_Ranking')

print(f"\\nTop 10 features by RFE ranking:")
print(rfe_rankings.head(10))
"""),
        
        new_code_cell("""### 5.3 Feature Selection Comparison

import matplotlib.pyplot as plt

# Compare feature sets
chi2_set = set(chi2_features)
rfe_set = set(rfe_features)

# Feature importance comparison (top 15)
top_n = 15
chi2_top = chi2_scores.head(top_n).set_index('Feature')['Chi2_Score']

# Normalize scores for comparison
chi2_top_norm = (chi2_top - chi2_top.min()) / (chi2_top.max() - chi2_top.min())

# Plot
fig, ax = plt.subplots(figsize=(12, 8))
x = range(top_n)
ax.barh(x, chi2_top_norm.values, alpha=0.6, label='Chi-square (normalized)', color='#3498db')
ax.set_yticks(x)
ax.set_yticklabels(chi2_top.index, fontsize=9)
ax.set_xlabel('Normalized Score', fontsize=12)
ax.set_title(f'Top {top_n} Features by Chi-square', fontsize=14, fontweight='bold')
ax.legend()
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# Print overlap statistics
overlap = chi2_set.intersection(rfe_set)
print(f"\\n📊 Feature Selection Comparison:")
print(f"Chi-square selected: {len(chi2_set)} features")
print(f"RFE selected: {len(rfe_set)} features")
print(f"Overlap: {len(overlap)} features ({len(overlap)/k_features*100:.1f}%)")
print(f"\\nCommon features: {list(overlap)[:10]}" + ("..." if len(overlap) > 10 else ""))
"""),
        
        new_code_cell("""### 5.4 Create Feature Sets for Model Training

# Create dictionary of feature sets for comprehensive comparison
feature_sets = {
    'all_features': X_train.columns.tolist(),
    'chi2_features': chi2_features,
    'rfe_features': rfe_features,
    'common_features': list(chi2_set.intersection(rfe_set))
}

print("\\n📦 Feature Sets Created:")
for name, features in feature_sets.items():
    print(f"{name}: {len(features)} features")

# This will be used for training models with different feature subsets
print("\\n✅ Ready for comprehensive model comparison with:")
print(f"   - {len(feature_sets)} feature selection strategies")
print(f"   - Multiple sampling strategies (Original, SMOTE, Undersampling)")
print(f"   - Multiple models (Logistic Regression, Random Forest, XGBoost)")
""")
    ]
    
    # Insert before model training
    insert_idx = model_idx - 1
    for i, cell in enumerate(cells_to_add):
        nb.cells.insert(insert_idx + i, cell)
    
    print("✅ Feature Selection methods added successfully")
    return True

def add_comprehensive_comparison(nb):
    """Add comprehensive model comparison framework."""
    print("Adding comprehensive model comparison framework...")
    
    # Add at the end
    insert_idx = len(nb.cells)
    
    cells_to_add = [
        new_markdown_cell("""## 6. Comprehensive Model Comparison

Now we'll train and compare all combinations of:
- **3 Sampling Strategies**: Original, BorderlineSMOTE, Random Undersampling
- **4 Feature Sets**: All features, Chi-square, RFE, Common features
- **3 Models**: Logistic Regression, Random Forest, XGBoost

**Total: 36 model configurations**
"""),
        
        new_code_cell("""### 6.1 Train All Model Configurations

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    accuracy_score, roc_auc_score, 
    average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='logloss')
}

# Define sampling strategies
sampling_strategies = {
    'Original': (X_train, y_train),
}

# Add SMOTE if available
if 'X_train_smote' in locals() and 'y_train_smote' in locals():
    sampling_strategies['BorderlineSMOTE'] = (X_train_smote, y_train_smote)

# Add Random Undersampling
if 'X_train_rus' in locals() and 'y_train_rus' in locals():
    sampling_strategies['Random Undersampling'] = (X_train_rus, y_train_rus)

# Results storage
results = []

print("="*70)
print("TRAINING ALL MODEL CONFIGURATIONS")
print("="*70)
print(f"\\nTotal configurations: {len(sampling_strategies)} × {len(feature_sets)} × {len(models)} = {len(sampling_strategies) * len(feature_sets) * len(models)}")
print("\\nThis may take a few minutes...\\n")

# Train all combinations
for sampling_name, (X_samp, y_samp) in sampling_strategies.items():
    for feature_name, features in feature_sets.items():
        # Select features
        X_train_subset = X_samp[features] if isinstance(X_samp, pd.DataFrame) else pd.DataFrame(X_samp, columns=X_train.columns)[features]
        X_val_subset = X_val[features]
        
        for model_name, model in models.items():
            try:
                # Train model
                model.fit(X_train_subset, y_samp)
                
                # Predict
                y_pred = model.predict(X_val_subset)
                y_pred_proba = model.predict_proba(X_val_subset)[:, 1]
                
                # Calculate metrics
                precision = precision_score(y_val, y_pred, zero_division=0)
                recall = recall_score(y_val, y_pred, zero_division=0)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                accuracy = accuracy_score(y_val, y_pred)
                roc_auc = roc_auc_score(y_val, y_pred_proba)
                pr_auc = average_precision_score(y_val, y_pred_proba)
                
                # Store results
                results.append({
                    'Sampling': sampling_name,
                    'Features': feature_name,
                    'Model': model_name,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'Accuracy': accuracy,
                    'ROC-AUC': roc_auc,
                    'PR-AUC': pr_auc,
                    'Train_Size': len(y_samp),
                    'Num_Features': len(features)
                })
                
                print(f"✓ {sampling_name:20s} | {feature_name:20s} | {model_name:20s} | F1: {f1:.3f}")
                
            except Exception as e:
                print(f"✗ {sampling_name:20s} | {feature_name:20s} | {model_name:20s} | Error: {str(e)[:50]}")

print(f"\\n✅ Training complete! {len(results)} configurations trained successfully.")
"""),
        
        new_code_cell("""### 6.2 Results Analysis

# Create results DataFrame
results_df = pd.DataFrame(results)

# Display top 10 configurations by F1-Score
print("="*70)
print("TOP 10 MODEL CONFIGURATIONS (by F1-Score)")
print("="*70)
print(results_df.nlargest(10, 'F1-Score')[['Sampling', 'Features', 'Model', 'F1-Score', 'ROC-AUC', 'Precision', 'Recall']])

# Display top 10 by ROC-AUC
print("\\n" + "="*70)
print("TOP 10 MODEL CONFIGURATIONS (by ROC-AUC)")
print("="*70)
print(results_df.nlargest(10, 'ROC-AUC')[['Sampling', 'Features', 'Model', 'ROC-AUC', 'F1-Score', 'Precision', 'Recall']])

# Best overall model
best_model = results_df.loc[results_df['F1-Score'].idxmax()]
print("\\n" + "="*70)
print("🏆 BEST MODEL CONFIGURATION")
print("="*70)
print(f"Sampling Strategy: {best_model['Sampling']}")
print(f"Feature Set: {best_model['Features']}")
print(f"Model: {best_model['Model']}")
print(f"\\nPerformance Metrics:")
print(f"  F1-Score:  {best_model['F1-Score']:.4f}")
print(f"  ROC-AUC:   {best_model['ROC-AUC']:.4f}")
print(f"  Precision: {best_model['Precision']:.4f}")
print(f"  Recall:    {best_model['Recall']:.4f}")
print(f"  Accuracy:  {best_model['Accuracy']:.4f}")
print(f"\\nDataset Info:")
print(f"  Training samples: {best_model['Train_Size']}")
print(f"  Number of features: {best_model['Num_Features']}")
"""),
        
        new_code_cell("""### 6.3 Performance Visualizations

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 1. F1-Score by Sampling Strategy
results_df.groupby('Sampling')['F1-Score'].mean().sort_values().plot(
    kind='barh', ax=axes[0, 0], color='#3498db'
)
axes[0, 0].set_title('Average F1-Score by Sampling Strategy', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('F1-Score', fontsize=12)

# 2. F1-Score by Feature Set
results_df.groupby('Features')['F1-Score'].mean().sort_values().plot(
    kind='barh', ax=axes[0, 1], color='#2ecc71'
)
axes[0, 1].set_title('Average F1-Score by Feature Set', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('F1-Score', fontsize=12)

# 3. F1-Score by Model
results_df.groupby('Model')['F1-Score'].mean().sort_values().plot(
    kind='barh', ax=axes[1, 0], color='#e74c3c'
)
axes[1, 0].set_title('Average F1-Score by Model', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('F1-Score', fontsize=12)

# 4. ROC-AUC vs F1-Score scatter
for model in results_df['Model'].unique():
    model_data = results_df[results_df['Model'] == model]
    axes[1, 1].scatter(model_data['ROC-AUC'], model_data['F1-Score'], 
                      label=model, alpha=0.6, s=100)
axes[1, 1].set_xlabel('ROC-AUC', fontsize=12)
axes[1, 1].set_ylabel('F1-Score', fontsize=12)
axes[1, 1].set_title('ROC-AUC vs F1-Score by Model', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Heatmap of F1-Scores
pivot_table = results_df.pivot_table(
    values='F1-Score', 
    index=['Sampling', 'Features'], 
    columns='Model'
)

plt.figure(figsize=(12, 10))
import seaborn as sns
sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn', 
            cbar_kws={'label': 'F1-Score'}, linewidths=0.5)
plt.title('F1-Score Heatmap: All Configurations', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Sampling Strategy | Feature Set', fontsize=12)
plt.tight_layout()
plt.show()
""")
    ]
    
    # Insert cells at the end
    for i, cell in enumerate(cells_to_add):
        nb.cells.insert(insert_idx + i, cell)
    
    print("✅ Comprehensive comparison framework added successfully")
    return True

def main():
    """Main function to enhance the notebook."""
    notebook_path = 'Load-Default_Prediction.ipynb'
    
    print("="*70)
    print("NOTEBOOK ENHANCEMENT SCRIPT")
    print("="*70)
    print(f"Target notebook: {notebook_path}\n")
    
    try:
        # Read notebook
        print("📖 Reading notebook...")
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        print(f"✅ Notebook loaded: {len(nb.cells)} cells\n")
        
        # Add enhancements
        success = True
        success &= add_random_undersampling(nb)
        success &= add_feature_selection(nb)
        success &= add_comprehensive_comparison(nb)
        
        if not success:
            print("\n❌ Some enhancements failed. Please review the output above.")
            return 1
        
        # Save enhanced notebook
        print(f"\n💾 Saving enhanced notebook...")
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        print(f"✅ Notebook saved: {len(nb.cells)} cells")
        print("\n" + "="*70)
        print("🎉 ENHANCEMENT COMPLETE!")
        print("="*70)
        print("\nAdded:")
        print("  ✓ Random Undersampling (second sampling strategy)")
        print("  ✓ Chi-square feature selection")
        print("  ✓ RFE feature selection")
        print("  ✓ Comprehensive model comparison (36 configurations)")
        print("\nNext steps:")
        print("  1. Open the notebook in Jupyter Lab")
        print("  2. Run all cells to execute the new code")
        print("  3. Review the results and visualizations")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
