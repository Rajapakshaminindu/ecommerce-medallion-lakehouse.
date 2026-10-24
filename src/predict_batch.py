"""
Batch Inference & Customer Retention Prioritization Engine
Applies trained Churn and LTV models across the entire Gold Feature Store.
Computes retention priority scores (Revenue at Risk) and generates actionable CRM intervention lists.
"""

import os
import sys
import argparse
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import joblib

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class BatchScoringEngine:
    """
    Orchestrates batch scoring across all customer feature vectors in the Gold Layer.
    Combines Churn Probability with Predicted LTV to calculate Expected Revenue at Risk.
    """

    def __init__(self, gold_dir: str = None, models_dir: str = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.gold_dir = gold_dir or os.path.join(base_dir, "data", "lakehouse", "gold")
        self.models_dir = models_dir or os.path.join(base_dir, "models")
        self.churn_pipeline = None
        self.ltv_pipeline = None

    def load_models(self):
        """Loads serialized Churn Classification and LTV Regression pipelines."""
        churn_model_path = os.path.join(self.models_dir, "churn_prediction_pipeline.joblib")
        ltv_model_path = os.path.join(self.models_dir, "ltv_prediction_pipeline.joblib")

        if not os.path.exists(churn_model_path):
            raise FileNotFoundError(
                f"Churn model not found at {churn_model_path}. Train models first using run_pipeline.py."
            )
        if not os.path.exists(ltv_model_path):
            raise FileNotFoundError(
                f"LTV model not found at {ltv_model_path}. Train models first using run_pipeline.py."
            )

        self.churn_pipeline = joblib.load(churn_model_path)
        self.ltv_pipeline = joblib.load(ltv_model_path)

    def run_batch_scoring(self, df_features: pd.DataFrame = None, threshold: float = 0.50) -> pd.DataFrame:
        """
        Executes batch scoring on customer feature store:
        1. Predicts churn probability and binary churn classification.
        2. Predicts customer lifetime value (LTV).
        3. Computes revenue at risk and assigns actionable risk segments.
        """
        if self.churn_pipeline is None or self.ltv_pipeline is None:
            self.load_models()

        if df_features is None:
            feature_store_path = os.path.join(self.gold_dir, "gold_customer_ml_feature_store.parquet")
            if not os.path.exists(feature_store_path):
                raise FileNotFoundError(f"Gold feature store not found at {feature_store_path}.")
            df_features = pd.read_parquet(feature_store_path)

        df_scored = df_features.copy()

        # Features required for Churn Classifier
        churn_num_features = [
            "frequency", "total_monetary_spend", "avg_order_value",
            "total_units_purchased", "return_rate_pct", "recency_days",
            "customer_tenure_days", "customer_age"
        ]
        churn_cat_features = ["region", "loyalty_tier"]
        X_churn = df_scored[churn_num_features + churn_cat_features]

        # Features required for LTV Regressor
        ltv_num_features = [
            "frequency", "avg_order_value", "total_units_purchased",
            "return_rate_pct", "recency_days", "customer_tenure_days", "customer_age"
        ]
        ltv_cat_features = ["region", "loyalty_tier"]
        X_ltv = df_scored[ltv_num_features + ltv_cat_features]

        # 1. Churn Predictions
        churn_probs = self.churn_pipeline.predict_proba(X_churn)[:, 1]
        df_scored["churn_risk_probability"] = np.round(churn_probs, 4)
        df_scored["predicted_churn"] = (df_scored["churn_risk_probability"] >= threshold).astype(int)

        # 2. LTV Predictions
        raw_ltv_preds = self.ltv_pipeline.predict(X_ltv)
        df_scored["predicted_ltv"] = np.round(np.clip(raw_ltv_preds, 0.0, None), 2)

        # 3. Revenue at Risk = Churn Probability * Predicted LTV
        df_scored["revenue_at_risk"] = np.round(
            df_scored["churn_risk_probability"] * df_scored["predicted_ltv"], 2
        )

        # 4. Actionable Risk Tiers
        conditions = [
            (df_scored["churn_risk_probability"] >= 0.60),
            (df_scored["churn_risk_probability"] >= 0.35) & (df_scored["churn_risk_probability"] < 0.60),
            (df_scored["churn_risk_probability"] < 0.35)
        ]
        tier_labels = ["Critical Risk", "Moderate Risk", "Low Risk"]
        df_scored["risk_tier"] = np.select(conditions, tier_labels, default="Low Risk")

        # 5. Targeted CRM Action Recommendations
        def assign_action(row):
            if row["churn_risk_probability"] >= 0.60 and row["predicted_ltv"] >= 500:
                return "Concierge VIP Outreach + 20% Loyalty Retention Pass"
            elif row["churn_risk_probability"] >= 0.60:
                return "Automated 15% Win-Back Discount Email"
            elif row["churn_risk_probability"] >= 0.35 and row["loyalty_tier"] in ["Gold", "Platinum"]:
                return "Dedicated Account Manager Check-in & Loyalty Point Multiplier"
            elif row["churn_risk_probability"] >= 0.35:
                return "Re-engagement Push Notification with Trending Catalog"
            elif row["loyalty_tier"] in ["Gold", "Platinum"]:
                return "Exclusive Early Access & Premium Upsell Campaign"
            else:
                return "Standard Marketing Cadence & Cross-sell"

        df_scored["recommended_action"] = df_scored.apply(assign_action, axis=1)
        df_scored["_scored_at"] = datetime.now(timezone.utc).isoformat()

        # Sort descending by revenue at risk
        df_scored = df_scored.sort_values(by="revenue_at_risk", ascending=False).reset_index(drop=True)

        # Persist to Gold Layer
        gold_predictions_path = os.path.join(self.gold_dir, "gold_customer_churn_predictions.parquet")
        df_scored.to_parquet(gold_predictions_path, index=False)
        print(f"[Gold Output] Persisted {len(df_scored)} scored customer profiles -> {gold_predictions_path}")

        return df_scored

    def export_retention_action_list(self, df_scored: pd.DataFrame, csv_path: str = None) -> str:
        """Exports high/moderate risk customer list as CSV for CRM/marketing ingestion."""
        if csv_path is None:
            csv_path = os.path.join(self.gold_dir, "retention_action_list.csv")

        action_cols = [
            "customer_id", "loyalty_tier", "region", "total_monetary_spend",
            "recency_days", "churn_risk_probability", "predicted_ltv",
            "revenue_at_risk", "risk_tier", "recommended_action"
        ]
        export_df = df_scored[df_scored["churn_risk_probability"] >= 0.35][action_cols]
        export_df.to_csv(csv_path, index=False)
        print(f"[Export] Saved {len(export_df)} actionable retention targets to CSV -> {csv_path}")
        return csv_path


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Batch Customer Churn & LTV Scoring Engine with Retention Prioritization"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Probability classification threshold for churn flagging (default: 0.50)"
    )
    parser.add_argument(
        "--top-at-risk",
        type=int,
        default=10,
        help="Display top N customers at highest revenue risk in terminal (default: 10)"
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Custom export path for retention campaign action list CSV"
    )
    parser.add_argument(
        "--gold-dir",
        type=str,
        default=None,
        help="Custom Gold data lakehouse directory path"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Custom models artifact directory path"
    )
    return parser.parse_args(args)


def main():
    args = parse_args()
    engine = BatchScoringEngine(gold_dir=args.gold_dir, models_dir=args.models_dir)
    print("\n" + "=" * 65)
    print(" === BATCH CUSTOMER CHURN & RETENTION SCORING ENGINE === ")
    print("=" * 65)

    df_scored = engine.run_batch_scoring(threshold=args.threshold)
    engine.export_retention_action_list(df_scored, csv_path=args.export_csv)

    total_customers = len(df_scored)
    critical_risk = (df_scored["risk_tier"] == "Critical Risk").sum()
    moderate_risk = (df_scored["risk_tier"] == "Moderate Risk").sum()
    total_rev_at_risk = df_scored["revenue_at_risk"].sum()

    print("\n" + "-" * 65)
    print(f"Total Customers Scored  : {total_customers}")
    print(f"Critical Churn Risk (>=60%): {critical_risk} ({critical_risk / total_customers * 100:.1f}%)")
    print(f"Moderate Risk (35-60%)   : {moderate_risk} ({moderate_risk / total_customers * 100:.1f}%)")
    print(f"Total Revenue at Risk   : ${total_rev_at_risk:,.2f}")
    print("-" * 65)

    print(f"\nTop {args.top_at_risk} Highest Revenue-at-Risk Customers:")
    preview_cols = ["customer_id", "loyalty_tier", "churn_risk_probability", "predicted_ltv", "revenue_at_risk", "risk_tier"]
    print(df_scored[preview_cols].head(args.top_at_risk).to_string(index=False))
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
