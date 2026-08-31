"""
Automated Pipeline & Data Quality Unit Tests
Ensures correctness across Bronze, Silver, Gold layers and ML inference.
"""

import os
import unittest
import pandas as pd
import numpy as np
from src.pipeline_bronze import BronzePipeline
from src.pipeline_silver import SilverPipeline
from src.pipeline_gold import GoldPipeline
from src.ml_model import CustomerChurnMLModel
from src.data_generator import generate_synthetic_dataset

class TestMedallionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.raw_dir = os.path.join(cls.base_dir, "data", "raw_source")
        cls.bronze_dir = os.path.join(cls.base_dir, "data", "lakehouse", "bronze")
        cls.silver_dir = os.path.join(cls.base_dir, "data", "lakehouse", "silver")
        cls.gold_dir = os.path.join(cls.base_dir, "data", "lakehouse", "gold")
        cls.models_dir = os.path.join(cls.base_dir, "models")

        # Generate sample data and run full pipeline once for tests
        generate_synthetic_dataset(cls.raw_dir, num_customers=50, num_orders=300, seed=123)
        BronzePipeline(cls.raw_dir, cls.bronze_dir).run()
        SilverPipeline(cls.bronze_dir, cls.silver_dir).run()
        GoldPipeline(cls.silver_dir, cls.gold_dir).run()
        cls.ml = CustomerChurnMLModel(cls.gold_dir, cls.models_dir)
        cls.ml.train_and_evaluate()

    def test_bronze_metadata_exists(self):
        """Test Bronze layer attaches required audit columns."""
        bronze_orders_path = os.path.join(self.bronze_dir, "bronze_orders.parquet")
        self.assertTrue(os.path.exists(bronze_orders_path))
        df = pd.read_parquet(bronze_orders_path)
        self.assertIn("_ingested_at", df.columns)
        self.assertIn("_source_file", df.columns)
        self.assertIn("_batch_id", df.columns)

    def test_silver_data_quality_rules(self):
        """Test Silver layer enforces deduplication and filters out invalid quantities."""
        silver_orders_path = os.path.join(self.silver_dir, "silver_orders_enriched.parquet")
        self.assertTrue(os.path.exists(silver_orders_path))
        df = pd.read_parquet(silver_orders_path)

        # 1. No duplicates in order_id
        self.assertEqual(df["order_id"].nunique(), len(df))

        # 2. No negative or zero quantities
        self.assertTrue((df["quantity"] > 0).all())

        # 3. Valid calculated financial fields
        self.assertTrue((df["gross_amount"] >= df["net_revenue"]).all())
        self.assertIn("profit", df.columns)
        self.assertIn("category", df.columns)

    def test_gold_kpi_aggregations(self):
        """Test Gold layer produces valid business aggregations."""
        daily_kpi_path = os.path.join(self.gold_dir, "gold_kpi_daily_revenue.parquet")
        self.assertTrue(os.path.exists(daily_kpi_path))
        df_daily = pd.read_parquet(daily_kpi_path)
        self.assertGreater(len(df_daily), 0)
        self.assertTrue((df_daily["net_revenue"] >= 0).all())

        feature_store_path = os.path.join(self.gold_dir, "gold_customer_ml_feature_store.parquet")
        self.assertTrue(os.path.exists(feature_store_path))
        df_rfm = pd.read_parquet(feature_store_path)
        self.assertIn("recency_days", df_rfm.columns)
        self.assertIn("frequency", df_rfm.columns)
        self.assertIn("total_monetary_spend", df_rfm.columns)
        self.assertIn("is_churn_risk", df_rfm.columns)

    def test_ml_model_prediction(self):
        """Test Machine Learning model loads and predicts valid probability."""
        sample_customer = {
            "frequency": 3,
            "total_monetary_spend": 450.0,
            "avg_order_value": 150.0,
            "total_units_purchased": 5,
            "return_rate_pct": 0.0,
            "recency_days": 45,
            "customer_tenure_days": 200,
            "customer_age": 34,
            "region": "Western",
            "loyalty_tier": "Silver"
        }
        res = self.ml.predict_single_customer(sample_customer)
        self.assertIn("churn_risk_probability", res)
        self.assertGreaterEqual(res["churn_risk_probability"], 0.0)
        self.assertLessEqual(res["churn_risk_probability"], 1.0)
        self.assertIn(res["is_churn_risk"], [0, 1])

if __name__ == "__main__":
    unittest.main()
