"""
Machine Learning Model Engine
Trains and evaluates predictive models (Customer Churn Risk & Customer Lifetime Value LTV) directly from the
Gold Layer ML Feature Store. Exports artifacts and feature importances for live inference.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error
)
import joblib

class CustomerChurnMLModel:
    def __init__(self, gold_dir: str, model_output_dir: str):
        self.gold_dir = gold_dir
        self.model_output_dir = model_output_dir
        os.makedirs(self.model_output_dir, exist_ok=True)
        self.pipeline = None
        self.metrics = {}
        self.feature_names = []

    def train_and_evaluate(self):
        # 1. Load Gold Feature Store
        src_path = os.path.join(self.gold_dir, "gold_customer_ml_feature_store.parquet")
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Feature store not found at {src_path}. Run Gold pipeline first.")
        
        df = pd.read_parquet(src_path)
        print(f"[INFO] Training ML model on {len(df)} customer feature vectors from Gold layer...")

        # 2. Define Features & Target
        numeric_features = [
            "frequency", "total_monetary_spend", "avg_order_value",
            "total_units_purchased", "return_rate_pct", "recency_days",
            "customer_tenure_days", "customer_age"
        ]
        categorical_features = ["region", "loyalty_tier"]
        target_col = "is_churn_risk"

        X = df[numeric_features + categorical_features]
        y = df[target_col]

        # 3. Train / Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        # 4. Preprocessing Pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
            ]
        )

        # 5. Model Architecture (Random Forest + Preprocessing)
        classifier = RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_split=4, random_state=42
        )

        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        # 6. Cross-Validation
        cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=5, scoring="roc_auc")
        print(f"[CV] 5-Fold Cross-Validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # 7. Model Training & Testing
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]

        self.metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
            "cv_roc_auc_mean": round(cv_scores.mean(), 4),
            "cv_roc_auc_std": round(cv_scores.std(), 4),
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }

        # 8. Feature Importance Extraction
        ohe_features = self.pipeline.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(categorical_features)
        all_features = numeric_features + list(ohe_features)
        importances = self.pipeline.named_steps["classifier"].feature_importances_

        feature_importance_list = [
            {"feature": f, "importance": round(float(imp), 4)}
            for f, imp in sorted(zip(all_features, importances), key=lambda x: x[1], reverse=True)
        ]

        # 9. Save Artifacts
        model_path = os.path.join(self.model_output_dir, "churn_prediction_pipeline.joblib")
        joblib.dump(self.pipeline, model_path)

        metadata = {
            "metrics": self.metrics,
            "feature_importance": feature_importance_list,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features
        }
        with open(os.path.join(self.model_output_dir, "model_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"[SUCCESS] Model artifact saved to: {model_path}")
        print(f"[METRICS] Test Metrics: ROC-AUC={self.metrics['roc_auc']}, Accuracy={self.metrics['accuracy']}, F1={self.metrics['f1_score']}")
        return self.metrics

    def predict_single_customer(self, customer_dict: dict):
        """Live inference on a single customer dictionary."""
        if self.pipeline is None:
            model_path = os.path.join(self.model_output_dir, "churn_prediction_pipeline.joblib")
            self.pipeline = joblib.load(model_path)
            
        df_single = pd.DataFrame([customer_dict])
        prob = self.pipeline.predict_proba(df_single)[0, 1]
        pred = int(prob >= 0.5)
        return {"churn_risk_probability": round(float(prob), 4), "is_churn_risk": pred}

class CustomerLTVMLModel:
    def __init__(self, gold_dir: str, model_output_dir: str):
        self.gold_dir = gold_dir
        self.model_output_dir = model_output_dir
        os.makedirs(self.model_output_dir, exist_ok=True)
        self.pipeline = None
        self.metrics = {}
        self.feature_names = []

    def train_and_evaluate(self):
        # 1. Load Gold Feature Store
        src_path = os.path.join(self.gold_dir, "gold_customer_ml_feature_store.parquet")
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Feature store not found at {src_path}. Run Gold pipeline first.")

        df = pd.read_parquet(src_path)
        print(f"[INFO] Training LTV Regression ML model on {len(df)} customer feature vectors...")

        # 2. Features for LTV prediction
        numeric_features = [
            "frequency", "avg_order_value", "total_units_purchased",
            "return_rate_pct", "recency_days", "customer_tenure_days", "customer_age"
        ]
        categorical_features = ["region", "loyalty_tier"]
        target_col = "total_monetary_spend"

        X = df[numeric_features + categorical_features]
        y = df[target_col]

        # 3. Train / Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        # 4. Preprocessing Pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
            ]
        )

        # 5. Model Architecture (Random Forest Regressor)
        regressor = RandomForestRegressor(
            n_estimators=100, max_depth=6, min_samples_split=4, random_state=42
        )

        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])

        # 6. Cross-Validation (R2 scoring)
        cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=5, scoring="r2")
        print(f"[CV] 5-Fold Cross-Validation LTV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # 7. Model Training & Testing
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        self.metrics = {
            "r2_score": round(float(r2), 4),
            "mae": round(float(mae), 2),
            "rmse": round(float(rmse), 2),
            "cv_r2_mean": round(float(cv_scores.mean()), 4),
            "cv_r2_std": round(float(cv_scores.std()), 4),
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }

        # 8. Feature Importance Extraction
        ohe_features = self.pipeline.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(categorical_features)
        all_features = numeric_features + list(ohe_features)
        importances = self.pipeline.named_steps["regressor"].feature_importances_

        feature_importance_list = [
            {"feature": f, "importance": round(float(imp), 4)}
            for f, imp in sorted(zip(all_features, importances), key=lambda x: x[1], reverse=True)
        ]

        # 9. Save Artifacts
        model_path = os.path.join(self.model_output_dir, "ltv_prediction_pipeline.joblib")
        joblib.dump(self.pipeline, model_path)

        metadata = {
            "metrics": self.metrics,
            "feature_importance": feature_importance_list,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features
        }
        with open(os.path.join(self.model_output_dir, "ltv_model_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"[SUCCESS] LTV Model artifact saved to: {model_path}")
        print(f"[METRICS] LTV Test Metrics: R2={self.metrics['r2_score']}, MAE=${self.metrics['mae']}, RMSE=${self.metrics['rmse']}")
        return self.metrics

    def predict_single_customer_ltv(self, customer_dict: dict):
        """Live inference on a single customer dictionary returning predicted LTV."""
        if self.pipeline is None:
            model_path = os.path.join(self.model_output_dir, "ltv_prediction_pipeline.joblib")
            self.pipeline = joblib.load(model_path)
            
        df_single = pd.DataFrame([customer_dict])
        pred_val = self.pipeline.predict(df_single)[0]
        return {"predicted_ltv": round(float(max(0.0, pred_val)), 2)}

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    gold_path = os.path.join(base_dir, "data", "lakehouse", "gold")
    models_path = os.path.join(base_dir, "models")
    
    churn_ml = CustomerChurnMLModel(gold_path, models_path)
    churn_ml.train_and_evaluate()

    ltv_ml = CustomerLTVMLModel(gold_path, models_path)
    ltv_ml.train_and_evaluate()
