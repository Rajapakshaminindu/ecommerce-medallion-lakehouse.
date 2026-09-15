"""
Gold Layer Aggregation & ML Feature Store Engine
Builds curated, high-performance analytical KPI tables for executive dashboards
and an ML Feature Store for predictive modeling.
"""

import os
from datetime import datetime, timezone
import pandas as pd
import numpy as np

class GoldPipeline:
    def __init__(self, silver_dir: str, gold_dir: str):
        self.silver_dir = silver_dir
        self.gold_dir = gold_dir
        os.makedirs(self.gold_dir, exist_ok=True)

    def generate_daily_kpis(self, df_orders: pd.DataFrame):
        """Builds daily business performance KPI table."""
        df_orders["order_date_day"] = df_orders["order_date"].dt.date

        daily_kpis = df_orders.groupby("order_date_day").agg(
            total_orders=("order_id", "nunique"),
            total_quantity=("quantity", "sum"),
            gross_revenue=("gross_amount", "sum"),
            total_discounts=("discount_amount", "sum"),
            net_revenue=("net_revenue", "sum"),
            total_profit=("profit", "sum"),
            completed_orders=("order_status", lambda s: (s == "COMPLETED").sum()),
            returned_orders=("order_status", lambda s: (s == "RETURNED").sum()),
            unique_customers=("customer_id", "nunique")
        ).reset_index()

        daily_kpis["avg_order_value"] = daily_kpis["net_revenue"] / daily_kpis["total_orders"]
        daily_kpis["profit_margin_pct"] = (daily_kpis["total_profit"] / daily_kpis["net_revenue"]) * 100.0
        daily_kpis["return_rate_pct"] = (daily_kpis["returned_orders"] / daily_kpis["total_orders"]) * 100.0
        daily_kpis["_gold_updated_at"] = datetime.now(timezone.utc).isoformat()

        target_path = os.path.join(self.gold_dir, "gold_kpi_daily_revenue.parquet")
        daily_kpis.to_parquet(target_path, index=False)
        print(f"[Gold KPI] Daily Revenue: {len(daily_kpis)} days aggregated -> {target_path}")
        return daily_kpis

    def generate_category_kpis(self, df_orders: pd.DataFrame):
        """Builds category & product performance analytics table."""
        cat_kpis = df_orders.groupby("category").agg(
            total_orders=("order_id", "nunique"),
            total_units_sold=("quantity", "sum"),
            net_revenue=("net_revenue", "sum"),
            total_profit=("profit", "sum"),
            returned_units=("order_status", lambda s: (s == "RETURNED").sum())
        ).reset_index()

        cat_kpis["profit_margin_pct"] = (cat_kpis["total_profit"] / cat_kpis["net_revenue"]) * 100.0
        cat_kpis["return_rate_pct"] = (cat_kpis["returned_units"] / cat_kpis["total_orders"]) * 100.0
        cat_kpis["_gold_updated_at"] = datetime.now(timezone.utc).isoformat()

        target_path = os.path.join(self.gold_dir, "gold_kpi_category_performance.parquet")
        cat_kpis.to_parquet(target_path, index=False)
        print(f"[Gold KPI] Category Performance: {len(cat_kpis)} categories -> {target_path}")
        return cat_kpis

    def generate_customer_ml_feature_store(self, df_orders: pd.DataFrame):
        """
        Builds an RFM (Recency, Frequency, Monetary) Customer Feature Store
        engineered specifically for Machine Learning model consumption.
        """
        max_date = df_orders["order_date"].max()

        # Group by customer
        cust_rfm = df_orders.groupby("customer_id").agg(
            last_order_date=("order_date", "max"),
            first_order_date=("order_date", "min"),
            frequency=("order_id", "nunique"),
            total_monetary_spend=("net_revenue", "sum"),
            total_profit_generated=("profit", "sum"),
            avg_order_value=("net_revenue", "mean"),
            total_units_purchased=("quantity", "sum"),
            total_returns=("order_status", lambda s: (s == "RETURNED").sum()),
            total_discounts_claimed=("discount_amount", "sum"),
            region=("region", "first"),
            loyalty_tier=("loyalty_tier", "first"),
            customer_age=("customer_age", "first")
        ).reset_index()

        # Recency (days since last purchase relative to lakehouse max snapshot date)
        cust_rfm["recency_days"] = (max_date - cust_rfm["last_order_date"]).dt.days
        cust_rfm["customer_tenure_days"] = (max_date - cust_rfm["first_order_date"]).dt.days
        cust_rfm["return_rate_pct"] = (cust_rfm["total_returns"] / cust_rfm["frequency"]) * 100.0

        # Define Machine Learning Targets:
        # 1. is_churn_risk: Recency > 120 days or high return rate without recent purchases
        cust_rfm["is_churn_risk"] = np.where(
            (cust_rfm["recency_days"] > 120) | (cust_rfm["return_rate_pct"] > 40.0), 1, 0
        )
        
        # 2. is_high_value: Top 25% in total spend
        spend_75th = cust_rfm["total_monetary_spend"].quantile(0.75)
        cust_rfm["is_high_value"] = np.where(cust_rfm["total_monetary_spend"] >= spend_75th, 1, 0)

        # 3. Customer Lifetime Value (LTV) metric
        cust_rfm["customer_ltv"] = cust_rfm["total_monetary_spend"].round(2)

        cust_rfm["_gold_updated_at"] = datetime.now(timezone.utc).isoformat()

        target_path = os.path.join(self.gold_dir, "gold_customer_ml_feature_store.parquet")
        cust_rfm.to_parquet(target_path, index=False)
        print(f"[Gold ML Feature Store] Customer Features: {len(cust_rfm)} customer vectors -> {target_path}")
        return cust_rfm

    def generate_cohort_retention_matrix(self, df_orders: pd.DataFrame):
        """
        Builds a monthly cohort retention matrix tracking customer retention over time.
        Cohorts are defined by customer acquisition month (first order).
        """
        df = df_orders.copy()
        customer_first_order = df.groupby("customer_id")["order_date"].min().reset_index()
        customer_first_order["cohort_month"] = customer_first_order["order_date"].dt.to_period("M")
        customer_first_order = customer_first_order[["customer_id", "cohort_month"]]

        df = df.merge(customer_first_order, on="customer_id", how="left")
        df["order_month"] = df["order_date"].dt.to_period("M")

        df["period_number"] = (
            (df["order_month"].dt.year - df["cohort_month"].dt.year) * 12 +
            (df["order_month"].dt.month - df["cohort_month"].dt.month)
        )

        cohort_data = df.groupby(["cohort_month", "period_number"])["customer_id"].nunique().reset_index()
        cohort_data.rename(columns={"customer_id": "active_customers"}, inplace=True)

        cohort_sizes = cohort_data[cohort_data["period_number"] == 0][["cohort_month", "active_customers"]]
        cohort_sizes.rename(columns={"active_customers": "cohort_size"}, inplace=True)

        cohort_df = cohort_data.merge(cohort_sizes, on="cohort_month", how="left")
        cohort_df["retention_rate_pct"] = round((cohort_df["active_customers"] / cohort_df["cohort_size"]) * 100.0, 2)
        cohort_df["cohort_month"] = cohort_df["cohort_month"].astype(str)
        cohort_df["_gold_updated_at"] = datetime.now(timezone.utc).isoformat()

        target_path = os.path.join(self.gold_dir, "gold_cohort_retention.parquet")
        cohort_df.to_parquet(target_path, index=False)
        print(f"[Gold KPI] Cohort Retention Matrix: {len(cohort_df)} cohort-period rows -> {target_path}")
        return cohort_df

    def run(self):
        print("\n==========================================")
        print(">>> RUNNING GOLD CURATION & FEATURE STORE LAYER")
        print("==========================================")
        orders_path = os.path.join(self.silver_dir, "silver_orders_enriched.parquet")
        df_orders = pd.read_parquet(orders_path)

        self.generate_daily_kpis(df_orders)
        self.generate_category_kpis(df_orders)
        self.generate_cohort_retention_matrix(df_orders)
        self.generate_customer_ml_feature_store(df_orders)
        print("[SUCCESS] Gold Layer Aggregations Complete.\n")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    silver_path = os.path.join(base_dir, "data", "lakehouse", "silver")
    gold_path = os.path.join(base_dir, "data", "lakehouse", "gold")
    
    pipeline = GoldPipeline(silver_path, gold_path)
    pipeline.run()
