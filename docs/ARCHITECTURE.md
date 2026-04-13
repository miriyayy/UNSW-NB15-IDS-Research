
---

# System Architecture

## Table of Contents
1. [High-Level System Overview](#high-level-system-overview)
2. [Data Pipeline Architecture](#data-pipeline-architecture)
3. [Preprocessing Module](#preprocessing-module)
4. [Feature Engineering Module](#feature-engineering-module)
5. [Model Training Pipeline](#model-training-pipeline)
6. [Inference Engine](#inference-engine)
7. [Evaluation Framework](#evaluation-framework)
8. [Deployment Architecture](#deployment-architecture)
9. [Technology Stack](#technology-stack)

---

## High-Level System Overview

The intrusion detection system follows a modular architecture with distinct stages for data ingestion, preprocessing, feature engineering, model training, and inference.

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNSW-NB15 RAW DATASET                         │
│                  (2.5M records, 49 features)                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DATA PREPROCESSING MODULE                       │
│  • Missing value imputation                                      │
│  • Feature removal (IPs, ports)                                  │
│  • Categorical encoding (one-hot, label)                         │
│  • Log transformation (skewed distributions)                     │
│  • MinMax normalization [0,1]                                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 FEATURE ENGINEERING MODULE                        │
│  • Correlation analysis                                          │
│  • Feature selection (10 methods)                                │
│  • Ratio features (byte_ratio, packet_ratio)                     │
│  • Temporal features (dur_category)                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING PIPELINE                        │
│                                                                   │
│  ┌─────────────────┐              ┌─────────────────┐           │
│  │  ML MODELS      │              │  DL MODELS      │           │
│  │  • XGBoost      │              │  • DNN Baseline │           │
│  │  • Random Forest│              │  • BatchNorm DNN│           │
│  │  • CatBoost     │              │  • Early Stop   │           │
│  │  • Decision Tree│              │  • Focal Loss   │           │
│  │  • Logistic Reg │              │  • Autoencoder  │           │
│  └─────────────────┘              └─────────────────┘           │
│                                                                   │
│  • 5-Fold Stratified Cross-Validation                            │
│  • GridSearchCV / RandomizedSearchCV                             │
│  • Class weighting / Focal Loss                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION FRAMEWORK                           │
│  • Confusion Matrix                                              │
│  • Accuracy, Precision, Recall, F1-Score                         │
│  • ROC-AUC curves                                                │
│  • Training time analysis                                        │
│  • Inference time analysis (µs/sample)                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      INFERENCE ENGINE                             │
│  • Real-time prediction API                                      │
│  • Model serving (XGBoost: 0.60 µs/sample)                       │
│  • Threshold-based alerting                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Pipeline Architecture

### 1. Data Ingestion

```
┌────────────────────┐
│  UNSW-NB15 Dataset │
│  ├─ training.csv   │ ──┐
│  └─ testing.csv    │   │
└────────────────────┘   │
                         │
                         ▼
              ┌──────────────────┐
              │  Data Loader     │
              │  • pandas.read_csv()
              │  • dtype validation
              │  • initial checks
              └─────────┬────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  Data Validation │
              │  • Schema check  │
              │  • Null counts   │
              │  • Class balance │
              └──────────────────┘
```

**Implementation:**

```python
import pandas as pd

def load_data(train_path, test_path):
    """Load and validate UNSW-NB15 dataset."""
    
    # Load data
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Validate schema
    assert df_train.shape[1] == 49, "Invalid number of features"
    assert 'Label' in df_train.columns, "Missing target variable"
    
    # Check class distribution
    print(f"Training set class distribution:\n{df_train['Label'].value_counts(normalize=True)}")
    
    return df_train, df_test
```

---

## Preprocessing Module

The preprocessing module transforms raw network traffic data into ML-ready feature matrices.

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                   PREPROCESSING MODULE                        │
│                                                               │
│  ┌─────────────────┐                                         │
│  │ Input Validator │                                         │
│  │ • Check dtypes  │                                         │
│  │ • Detect nulls  │                                         │
│  └────────┬────────┘                                         │
│           │                                                   │
│           ▼                                                   │
│  ┌─────────────────┐       ┌─────────────────┐              │
│  │ Feature Dropper │       │ Missing Value   │              │
│  │ • srcip, dstip  │       │ Imputer         │              │
│  │ • sport, dsport │──────▶│ • Median (num)  │              │
│  │ • attack_cat    │       │ • Mode (cat)    │              │
│  └─────────────────┘       └────────┬────────┘              │
│                                     │                         │
│                                     ▼                         │
│                            ┌─────────────────┐               │
│                            │ Categorical     │               │
│                            │ Encoder         │               │
│                            │ • Label: state  │               │
│                            │ • OneHot: proto │               │
│                            │   service, etc. │               │
│                            └────────┬────────┘               │
│                                     │                         │
│                                     ▼                         │
│                            ┌─────────────────┐               │
│                            │ Log Transform   │               │
│                            │ • sbytes        │               │
│                            │ • dbytes        │               │
│                            │ • Sload, Dload  │               │
│                            └────────┬────────┘               │
│                                     │                         │
│                                     ▼                         │
│                            ┌─────────────────┐               │
│                            │ MinMax Scaler   │               │
│                            │ Range: [0, 1]   │               │
│                            └────────┬────────┘               │
│                                     │                         │
│                                     ▼                         │
│                            ┌─────────────────┐               │
│                            │ Preprocessed    │               │
│                            │ Feature Matrix  │               │
│                            │ Shape: (N, 72)  │               │
│                            └─────────────────┘               │
└──────────────────────────────────────────────────────────────┘
```

### Implementation

```python
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import numpy as np

class PreprocessingPipeline:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = MinMaxScaler()
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        
    def fit_transform(self, df):
        """Apply full preprocessing pipeline."""
        
        # Step 1: Drop high-cardinality features
        df = df.drop(['srcip', 'sport', 'dstip', 'dsport', 'attack_cat'], axis=1)
        
        # Step 2: Separate features by type
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target from feature lists
        num_cols.remove('Label')
        
        # Step 3: Impute missing values
        df[num_cols] = self.num_imputer.fit_transform(df[num_cols])
        df[cat_cols] = self.cat_imputer.fit_transform(df[cat_cols])
        
        # Step 4: Encode categorical features
        # Label encoding for ordinal
        for col in ['state']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        # One-hot encoding for nominal
        df = pd.get_dummies(df, columns=['proto', 'service'], drop_first=False)
        
        # Step 5: Log transformation
        skewed_features = ['sbytes', 'dbytes', 'Sload', 'Dload', 'dur']
        for col in skewed_features:
            df[col] = np.log1p(df[col])
        
        # Step 6: Normalize numerical features
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols.remove('Label')
        df[num_cols] = self.scaler.fit_transform(df[num_cols])
        
        return df
```

---

## Feature Engineering Module

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                 FEATURE ENGINEERING MODULE                    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              FEATURE SELECTION METHODS               │    │
│  │                                                       │    │
│  │  ┌───────────┐  ┌───────────┐  ┌──────────────┐    │    │
│  │  │  ANOVA    │  │  Chi²     │  │  Mutual Info │    │    │
│  │  │  F-Test   │  │  Test     │  │              │    │    │
│  │  └─────┬─────┘  └─────┬─────┘  └──────┬───────┘    │    │
│  │        └──────────────┴────────────────┘            │    │
│  │                       │                             │    │
│  │  ┌───────────┐  ┌─────▼─────┐  ┌──────────────┐    │    │
│  │  │  LASSO    │  │Extra Trees│  │  LightGBM    │    │    │
│  │  │  L1 Reg   │  │Classifier │  │  Import      │    │    │
│  │  └─────┬─────┘  └─────┬─────┘  └──────┬───────┘    │    │
│  │        └──────────────┴────────────────┘            │    │
│  │                       │                             │    │
│  │  ┌───────────┐  ┌─────▼─────┐  ┌──────────────┐    │    │
│  │  │ RF Import │  │  XGB      │  │  RFE         │    │    │
│  │  │ -ance     │  │  Import   │  │ (Recursive)  │    │    │
│  │  └─────┬─────┘  └─────┬─────┘  └──────┬───────┘    │    │
│  │        └──────────────┴────────────────┘            │    │
│  │                       │                             │    │
│  │                       ▼                             │    │
│  │              ┌────────────────┐                     │    │
│  │              │ Consensus Top  │                     │    │
│  │              │ 20 Features    │                     │    │
│  │              └────────────────┘                     │    │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            DERIVED FEATURE CREATION                  │    │
│  │                                                       │    │
│  │  • byte_ratio = sbytes / dbytes                      │    │
│  │  • packet_ratio = Spkts / Dpkts                      │    │
│  │  • load_imbalance = |Sload - Dload|                  │    │
│  │  • interpkt_variance = |Sintpkt - Dintpkt|           │    │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Implementation

```python
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

class FeatureEngineer:
    def __init__(self, n_features=20):
        self.n_features = n_features
        self.selected_features = None
        
    def select_features(self, X, y):
        """Apply multiple feature selection methods and find consensus."""
        
        feature_scores = {}
        
        # Method 1: ANOVA F-test
        selector = SelectKBest(f_classif, k=self.n_features)
        selector.fit(X, y)
        anova_features = X.columns[selector.get_support()].tolist()
        
        # Method 2: Chi-square
        selector = SelectKBest(chi2, k=self.n_features)
        selector.fit(X, y)
        chi2_features = X.columns[selector.get_support()].tolist()
        
        # Method 3: Mutual Information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_features = X.columns[np.argsort(mi_scores)[-self.n_features:]].tolist()
        
        # Method 4: Random Forest Importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_features = X.columns[np.argsort(rf.feature_importances_)[-self.n_features:]].tolist()
        
        # Method 5: XGBoost Importance
        xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        xgb_model.fit(X, y)
        xgb_features = X.columns[np.argsort(xgb_model.feature_importances_)[-self.n_features:]].tolist()
        
        # Consensus: Features appearing in most methods
        all_features = anova_features + chi2_features + mi_features + rf_features + xgb_features
        feature_counts = pd.Series(all_features).value_counts()
        self.selected_features = feature_counts.head(self.n_features).index.tolist()
        
        return self.selected_features
    
    def create_derived_features(self, df):
        """Create ratio and interaction features."""
        
        df['byte_ratio'] = df['sbytes'] / (df['dbytes'] + 1e-5)
        df['packet_ratio'] = df['Spkts'] / (df['Dpkts'] + 1e-5)
        df['load_imbalance'] = np.abs(df['Sload'] - df['Dload'])
        df['interpkt_variance'] = np.abs(df['Sintpkt'] - df['Dintpkt'])
        
        return df
```

---

## Model Training Pipeline

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   MODEL TRAINING PIPELINE                     │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         STRATIFIED K-FOLD CROSS-VALIDATION            │   │
│  │                    (K = 5)                             │   │
│  │                                                        │   │
│  │  Fold 1: ████████████████░░░░  Train | ░░ Valid       │   │
│  │  Fold 2: ░░░░████████████████░░  Train | ░░ Valid     │   │
│  │  Fold 3: ░░░░░░░░████████████████  Train | ░░ Valid   │   │
│  │  Fold 4: ████░░░░░░░░████████████  Train | ░░ Valid   │   │
│  │  Fold 5: ████████████░░░░░░░░████  Train | ░░ Valid   │   │
│  └─────────────────────┬────────────────────────────────┘   │
│                        │                                     │
│                        ▼                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              HYPERPARAMETER TUNING                   │   │
│  │                                                       │   │
│  │  ┌──────────────────┐     ┌──────────────────┐      │   │
│  │  │  GridSearchCV    │     │ RandomizedSearch │      │   │
│  │  │  • Exhaustive    │     │ • Sampling       │      │   │
│  │  │  • Best params   │     │ • Efficient      │      │   │
│  │  └──────────────────┘     └──────────────────┘      │   │
│  └─────────────────────┬────────────────────────────────┘   │
│                        │                                     │
│                        ▼                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            PARALLEL MODEL TRAINING                   │   │
│  │                                                       │   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐    │   │
│  │  │XGBoost │  │Random  │  │CatBoost│  │Decision│    │   │
│  │  │        │  │Forest  │  │        │  │Tree    │    │   │
│  │  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘    │   │
│  │      └───────────┴───────────┴───────────┘          │   │
│  │                        │                             │   │
│  │  ┌────────┐  ┌─────────▼┐  ┌────────┐  ┌────────┐  │   │
│  │  │DNN     │  │DNN +    │  │DNN +   │  │Auto-   │   │   │
│  │  │Baseline│  │BatchNrm │  │Focal   │  │encoder │   │   │
│  │  └───┬────┘  └───┬─────┘  └───┬────┘  └───┬────┘   │   │
│  │      └───────────┴────────────┴───────────┘         │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              MODEL PERFORMANCE LOGGING                │  │
│  │  For each fold:                                       │  │
│  │  • Accuracy, Precision, Recall, F1-Score              │  │
│  │  • Confusion Matrix                                   │  │
│  │  • ROC-AUC                                            │  │
│  │  • Training time                                      │  │
│  │  • Inference time (µs/sample)                         │  │
│  └────────────────────┬─────────────────────────────────┘  │
│                       │                                     │
│                       ▼                                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           AGGREGATE RESULTS & MODEL SELECTION         │  │
│  │  • Mean ± Std across folds                            │  │
│  │  • Select best model (highest F1-score)               │  │
│  │  • Save model artifacts                               │  │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### Implementation

```python
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

class ModelTrainer:
    def __init__(self, n_folds=5):
        self.n_folds = n_folds
        self.results = {}
        
    def train_with_cv(self, model, X, y, model_name):
        """Train model with stratified k-fold cross-validation."""
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        fold_results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'train_time': [],
            'pred_time': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Train
            start_train = time.time()
            model.fit(X_train_fold, y_train_fold)
            train_time = time.time() - start_train
            
            # Predict
            start_pred = time.time()
            y_pred = model.predict(X_val_fold)
            pred_time = (time.time() - start_pred) * 1e6 / len(X_val_fold)  # µs/sample
            
            # Metrics
            fold_results['accuracy'].append(accuracy_score(y_val_fold, y_pred))
            fold_results['precision'].append(precision_score(y_val_fold, y_pred))
            fold_results['recall'].append(recall_score(y_val_fold, y_pred))
            fold_results['f1'].append(f1_score(y_val_fold, y_pred))
            fold_results['train_time'].append(train_time)
            fold_results['pred_time'].append(pred_time)
        
        # Aggregate results
        self.results[model_name] = {
            'accuracy': np.mean(fold_results['accuracy']),
            'precision': np.mean(fold_results['precision']),
            'recall': np.mean(fold_results['recall']),
            'f1': np.mean(fold_results['f1']),
            'train_time': np.mean(fold_results['train_time']),
            'pred_time': np.mean(fold_results['pred_time'])
        }
        
        return self.results[model_name]
```

---

## Inference Engine

### Real-Time Prediction Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      INFERENCE ENGINE                         │
│                                                               │
│  ┌────────────────┐                                          │
│  │ Network Traffic│                                          │
│  │ Raw Packet     │                                          │
│  └───────┬────────┘                                          │
│          │                                                    │
│          ▼                                                    │
│  ┌────────────────┐                                          │
│  │ Feature        │                                          │
│  │ Extraction     │                                          │
│  │ • Parse packets│                                          │
│  │ • Flow stats   │                                          │
│  └───────┬────────┘                                          │
│          │                                                    │
│          ▼                                                    │
│  ┌────────────────┐                                          │
│  │ Preprocessing  │                                          │
│  │ • Encoding     │                                          │
│  │ • Scaling      │                                          │
│  │ • Transform    │                                          │
│  └───────┬────────┘                                          │
│          │                                                    │
│          ▼                                                    │
│  ┌────────────────┐         ┌──────────────┐                │
│  │ Model Serving  │────────▶│ Trained Model│                │
│  │ • Load model   │         │ (XGBoost)    │                │
│  │ • Batch predict│         │ 0.60 µs/samp │                │
│  └───────┬────────┘         └──────────────┘                │
│          │                                                    │
│          ▼                                                    │
│  ┌────────────────┐                                          │
│  │ Post-Processing│                                          │
│  │ • Threshold    │                                          │
│  │ • Confidence   │                                          │
│  └───────┬────────┘                                          │
│          │                                                    │
│          ▼                                                    │
│  ┌────────────────┐                                          │
│  │ Alert System   │                                          │
│  │ • If attack:   │                                          │
│  │   - Log event  │                                          │
│  │   - Notify SOC │                                          │
│  │   - Block IP   │                                          │
│  └────────────────┘                                          │
└──────────────────────────────────────────────────────────────┘
```

### Implementation

```python
import joblib
import pandas as pd

class InferenceEngine:
    def __init__(self, model_path, preprocessor_path):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        
    def predict(self, raw_traffic_data):
        """Make real-time prediction on network traffic."""
        
        # Step 1: Preprocess
        X = self.preprocessor.transform(raw_traffic_data)
        
        # Step 2: Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Step 3: Post-process
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'flow_id': i,
                'prediction': 'Attack' if pred == 1 else 'Normal',
                'confidence': prob,
                'alert': prob > 0.8  # High-confidence attack
            })
        
        return results
    
    def batch_predict(self, traffic_stream, batch_size=1000):
        """Process traffic in batches for efficiency."""
        
        for i in range(0, len(traffic_stream), batch_size):
            batch = traffic_stream[i:i+batch_size]
            yield self.predict(batch)
```

---

## Evaluation Framework

### Metric Computation Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                    EVALUATION FRAMEWORK                       │
│                                                               │
│  ┌────────────────┐       ┌────────────────┐                │
│  │ Ground Truth   │       │ Predictions    │                │
│  │ (y_true)       │       │ (y_pred)       │                │
│  └───────┬────────┘       └───────┬────────┘                │
│          └────────────┬───────────┘                          │
│                       │                                       │
│                       ▼                                       │
│          ┌─────────────────────────┐                         │
│          │   Confusion Matrix      │                         │
│          │  ┌────────┬────────┐    │                         │
│          │  │   TN   │   FP   │    │                         │
│          │  ├────────┼────────┤    │                         │
│          │  │   FN   │   TP   │    │                         │
│          │  └────────┴────────┘    │                         │
│          └────────────┬────────────┘                         │
│                       │                                       │
│          ┌────────────┴────────────┐                         │
│          │                         │                         │
│          ▼                         ▼                         │
│  ┌──────────────┐         ┌──────────────┐                  │
│  │ Classification│         │  ROC-AUC     │                  │
│  │ Metrics       │         │  Analysis    │                  │
│  │ • Accuracy    │         │ • FPR, TPR   │                  │
│  │ • Precision   │         │ • Thresholds │                  │
│  │ • Recall      │         │ • AUC score  │                  │
│  │ • F1-Score    │         │              │                  │
│  └───────────────┘         └──────────────┘                  │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              TIME COMPLEXITY ANALYSIS                 │   │
│  │  • Training Time: Σ(fit_time) across folds           │   │
│  │  • Inference Time: mean(predict_time) µs/sample       │   │
│  │  • Memory Usage: model.get_booster().get_dump()       │   │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## Deployment Architecture

### Production System

```
┌────────────────────────────────────────────────────────────────┐
│                    PRODUCTION DEPLOYMENT                        │
│                                                                 │
│  ┌──────────────┐          ┌──────────────┐                   │
│  │ Network      │          │ Packet       │                   │
│  │ Interface    │─────────▶│ Capture      │                   │
│  │ (eth0)       │          │ (tcpdump)    │                   │
│  └──────────────┘          └──────┬───────┘                   │
│                                    │                            │
│                                    ▼                            │
│                           ┌──────────────┐                     │
│                           │ Feature      │                     │
│                           │ Extractor    │                     │
│                           │ (Python)     │                     │
│                           └──────┬───────┘                     │
│                                  │                              │
│                                  ▼                              │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              KAFKA MESSAGE QUEUE                         │  │
│  │  Topic: "network_flows"                                  │  │
│  │  Partitions: 10                                          │  │
│  │  Retention: 7 days                                       │  │
│  └────────────────────┬────────────────────────────────────┘  │
│                       │                                        │
│                       ▼                                        │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │           PREDICTION MICROSERVICE CLUSTER                │  │
│  │                                                           │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐               │  │
│  │  │ Worker 1 │  │ Worker 2 │  │ Worker N │               │  │
│  │  │ XGBoost  │  │ XGBoost  │  │ XGBoost  │               │  │
│  │  │ Model    │  │ Model    │  │ Model    │               │  │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘               │  │
│  │       └─────────────┴─────────────┘                       │  │
│  └─────────────────────────────────────────────────────────┘  │
│                        │                                        │
│                        ▼                                        │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │               RESULT AGGREGATION                         │  │
│  │  • Predictions                                           │  │
│  │  • Confidence scores                                     │  │
│  │  • Alert flags                                           │  │
│  └────────────────────┬────────────────────────────────────┘  │
│                       │                                        │
│           ┌───────────┴───────────┐                           │
│           │                       │                           │
│           ▼                       ▼                           │
│  ┌──────────────┐       ┌──────────────┐                     │
│  │ PostgreSQL   │       │ Alerting     │                     │
│  │ Database     │       │ System       │                     │
│  │ • Flow logs  │       │ • Email      │                     │
│  │ • Predictions│       │ • Slack      │                     │
│  │ • Metrics    │       │ • SIEM       │                     │
│  └──────────────┘       └──────────────┘                     │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                 MONITORING DASHBOARD                     │  │
│  │  • Grafana visualization                                 │  │
│  │  • Prometheus metrics                                    │  │
│  │  • Attack rate over time                                 │  │
│  │  • Model performance drift detection                     │  │
│  └─────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### Core Technologies

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Data Processing** | Python 3.8+ | Primary language |
| | Pandas | Data manipulation |
| | NumPy | Numerical computation |
| **Machine Learning** | Scikit-learn | Classical ML algorithms |
| | XGBoost | Gradient boosting |
| | CatBoost | Categorical boosting |
| **Deep Learning** | TensorFlow 2.x | Neural network framework |
| | Keras | High-level API |
| **Feature Engineering** | SciPy | Statistical tests |
| | mrmr-selection | Feature selection |
| **Visualization** | Matplotlib | Static plots |
| | Seaborn | Statistical graphics |
| **Model Serving** | FastAPI | REST API |
| | Uvicorn | ASGI server |
| **Message Queue** | Apache Kafka | Stream processing |
| **Database** | PostgreSQL | Relational storage |
| **Monitoring** | Prometheus | Metrics collection |
| | Grafana | Visualization |
| **Containerization** | Docker | Application packaging |
| | Kubernetes | Orchestration |

### Development Environment

```yaml
# environment.yml
name: ids-unsw-nb15
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.8
  - pandas=1.3.3
  - numpy=1.21.2
  - scikit-learn=1.0.0
  - matplotlib=3.4.3
  - seaborn=0.11.2
  - scipy=1.7.1
  - pip:
    - xgboost==1.5.0
    - catboost==1.0.0
    - tensorflow==2.7.0
    - mrmr-selection==0.2.6
    - fastapi==0.70.0
    - uvicorn==0.15.0
```

---

## Summary

This architecture implements a production-ready intrusion detection system with:

✅ **Modular Design**: Separate preprocessing, feature engineering, training, and inference modules  
✅ **Scalability**: Kafka-based stream processing, microservice deployment  
✅ **Performance**: 0.60 µs/sample inference time (XGBoost)  
✅ **Robustness**: 5-fold cross-validation, class imbalance handling  
✅ **Monitoring**: Real-time performance tracking via Prometheus/Grafana  
✅ **Reproducibility**: Version-controlled pipelines, containerized deployment  

The system achieves **F1-score of 98.08%** with XGBoost while maintaining real-time processing capabilities suitable for production network security operations.
