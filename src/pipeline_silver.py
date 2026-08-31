"""
Silver Layer Transformation Engine
Cleanses, deduplicates, validates schemas, enforces data quality rules,
and enriches raw Bronze data into trusted, analytical tables.
"""

import os
from datetime import datetime, timezone
import pandas as pd
import numpy as np

class SilverPipeline:
    def __init__(self, bronze_dir: str, silver_dir: str):
        self.bronze_dir = bronze_dir
        self.silver_dir = silver_dir
        os.makedirs(self.silver_dir, exist_ok=True)
        self.data_quality_report = {}

    def transform_customers(self):
        """Cleanses customer dimension table."""
        src_path = os.path.join(self.bronze_dir, "bronze_customers.parquet")
        df = pd.read_parquet(src_path)

        # Standardize strings & dates
        df["customer_name"] = df["customer_name"].str.strip()
        df["region"] = df["region"].str.strip()
        df["loyalty_tier"] = df["loyalty_tier"].str.strip()
        df["signup_date"] = pd.to_datetime(df["signup_date"])
        
        # Add metadata
        df["_silver_processed_at"] = datetime.now(timezone.utc).isoformat()

        target_path = os.path.join(self.silver_dir, "silver_customers.parquet")
        df.to_parquet(target_path, index=False)
        print(f"[Silver Cleaned] Customers: {len(df)} records -> {target_path}")
        return df

    def transform_products(self):
        """Cleanses product master table."""
        src_path = os.path.join(self.bronze_dir, "bronze_products.parquet")
        df = pd.read_parquet(src_path)

        df["product_name"] = df["product_name"].str.strip()
        df["category"] = df["category"].str.strip()
        df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
        df["unit_cost"] = pd.to_numeric(df["unit_cost"], errors="coerce")
        
        # Calculate standard product margin
        df["unit_margin"] = df["unit_price"] - df["unit_cost"]
        df["_silver_processed_at"] = datetime.now(timezone.utc).isoformat()

        target_path = os.path.join(self.silver_dir, "silver_products.parquet")
        df.to_parquet(target_path, index=False)
        print(f"[Silver Cleaned] Products: {len(df)} records -> {target_path}")
        return df

    def transform_and_enrich_orders(self, df_customers: pd.DataFrame, df_products: pd.DataFrame):
        """
        Applies rigorous data quality rules, deduplication, and joins Bronze Orders
        with Customers and Products to produce Silver Enriched Orders.
        """
        src_path = os.path.join(self.bronze_dir, "bronze_orders.parquet")
        df = pd.read_parquet(src_path)
        initial_count = len(df)

        # 1. Deduplication
        df_dedup = df.drop_duplicates(subset=["order_id"], keep="first").copy()
        duplicates_removed = initial_count - len(df_dedup)

        # 2. String trimming & cleaning
        df_dedup["order_status"] = df_dedup["order_status"].astype(str).str.strip().str.upper()
        df_dedup["payment_method"] = df_dedup["payment_method"].fillna("Unknown").astype(str).str.strip()

        # 3. Data type coercion & date parsing
        df_dedup["order_date"] = pd.to_datetime(df_dedup["order_date"], errors="coerce")
        df_dedup["quantity"] = pd.to_numeric(df_dedup["quantity"], errors="coerce")
        df_dedup["unit_price"] = pd.to_numeric(df_dedup["unit_price"], errors="coerce")
        df_dedup["discount_pct"] = pd.to_numeric(df_dedup["discount_pct"], errors="coerce").fillna(0.0)

        # 4. Data Quality Filtering (Remove invalid transactions e.g. quantity <= 0)
        valid_mask = (
            (df_dedup["quantity"] > 0) &
            (df_dedup["unit_price"] > 0) &
            (df_dedup["order_date"].notnull())
        )
        invalid_rows_removed = int((~valid_mask).sum())
        df_valid = df_dedup[valid_mask].copy()

        # 5. Financial Metric Calculations
        df_valid["gross_amount"] = df_valid["quantity"] * df_valid["unit_price"]
        df_valid["discount_amount"] = df_valid["gross_amount"] * df_valid["discount_pct"]
        df_valid["net_revenue"] = df_valid["gross_amount"] - df_valid["discount_amount"]

        # 6. Relational Joins (Enrichment)
        # Join Customer details
        cust_cols = ["customer_id", "region", "loyalty_tier", "signup_date", "customer_age"]
        df_enriched = df_valid.merge(df_customers[cust_cols], on="customer_id", how="left")

        # Join Product details
        prod_cols = ["product_id", "product_name", "category", "unit_cost"]
        df_enriched = df_enriched.merge(df_products[prod_cols], on="product_id", how="left")

        # Calculate Total Cost and Profit
        df_enriched["total_cost"] = df_enriched["quantity"] * df_enriched["unit_cost"]
        df_enriched["profit"] = df_enriched["net_revenue"] - df_enriched["total_cost"]
        df_enriched["profit_margin_pct"] = (df_enriched["profit"] / df_enriched["net_revenue"]) * 100.0

        # Audit metadata
        df_enriched["_silver_processed_at"] = datetime.now(timezone.utc).isoformat()

        # 7. Save Silver Enriched Dataset
        target_path = os.path.join(self.silver_dir, "silver_orders_enriched.parquet")
        df_enriched.to_parquet(target_path, index=False)

        self.data_quality_report = {
            "initial_bronze_rows": initial_count,
            "duplicates_removed": duplicates_removed,
            "corrupted_rows_filtered": invalid_rows_removed,
            "clean_silver_rows": len(df_enriched),
            "quality_pass_rate_pct": round((len(df_enriched) / initial_count) * 100, 2)
        }

        print(f"[Silver Enriched] Orders: {len(df_enriched)} valid rows -> {target_path}")
        print(f"   [Data Quality Summary] {duplicates_removed} duplicates removed, {invalid_rows_removed} invalid rows filtered.")
        return df_enriched

    def run(self):
        print("\n==========================================")
        print(">>> RUNNING SILVER TRANSFORMATION LAYER")
        print("==========================================")
        df_cust = self.transform_customers()
        df_prod = self.transform_products()
        self.transform_and_enrich_orders(df_cust, df_prod)
        print("[SUCCESS] Silver Layer Transformation Complete.\n")
        return self.data_quality_report

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    bronze_path = os.path.join(base_dir, "data", "lakehouse", "bronze")
    silver_path = os.path.join(base_dir, "data", "lakehouse", "silver")
    
    pipeline = SilverPipeline(bronze_path, silver_path)
    pipeline.run()
