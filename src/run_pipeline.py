"""
Master Pipeline Orchestrator (ELT)
Executes the full Medallion Architecture workflow from Raw Ingestion to Machine Learning.
"""

import os
import sys
import time

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_generator import generate_synthetic_dataset
from src.pipeline_bronze import BronzePipeline
from src.pipeline_silver import SilverPipeline
from src.pipeline_gold import GoldPipeline
from src.ml_model import CustomerChurnMLModel

def run_full_lakehouse_pipeline(generate_fresh_data: bool = True):
    start_time = time.time()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    raw_dir = os.path.join(base_dir, "data", "raw_source")
    bronze_dir = os.path.join(base_dir, "data", "lakehouse", "bronze")
    silver_dir = os.path.join(base_dir, "data", "lakehouse", "silver")
    gold_dir = os.path.join(base_dir, "data", "lakehouse", "gold")
    models_dir = os.path.join(base_dir, "models")

    print("\n" + "=" * 65)
    print(" === E-COMMERCE MEDALLION LAKEHOUSE & ML PIPELINE RUNNER === ")
    print("=" * 65)

    # Step 0: Data Generation (Simulated Data Ingestion Stream)
    if generate_fresh_data or not os.path.exists(raw_dir) or len(os.listdir(raw_dir)) == 0:
        print("\n[Step 0/4] Generating Fresh Simulated Transaction Batches...")
        generate_synthetic_dataset(raw_dir, num_customers=300, num_orders=2000)

    # Step 1: Bronze Layer Ingestion
    print("\n[Step 1/4] Ingesting Raw Data into Bronze Layer (Parquet + Metadata)...")
    bronze = BronzePipeline(raw_dir, bronze_dir)
    bronze_summary = bronze.run()

    # Step 2: Silver Layer Transformation & Cleaning
    print("\n[Step 2/4] Executing Silver Data Cleansing, Deduplication & Enrichment...")
    silver = SilverPipeline(bronze_dir, silver_dir)
    silver_summary = silver.run()

    # Step 3: Gold Layer Curation & Feature Store
    print("\n[Step 3/4] Generating Gold Business KPI Tables & ML Feature Store...")
    gold = GoldPipeline(silver_dir, gold_dir)
    gold.run()

    # Step 4: Machine Learning Training & Evaluation
    print("\n[Step 4/4] Training Customer Churn ML Model from Gold Feature Store...")
    ml = CustomerChurnMLModel(gold_dir, models_dir)
    metrics = ml.train_and_evaluate()

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 65)
    print(f" [SUCCESS] PIPELINE EXECUTION COMPLETED IN {elapsed} SECONDS!")
    print("=" * 65)
    print(f" - Bronze Orders Ingested : {bronze_summary['orders']['orders']}")
    print(f" - Silver Cleaned Orders  : {silver_summary['clean_silver_rows']}")
    print(f" - Data Quality Pass Rate : {silver_summary['quality_pass_rate_pct']}%")
    print(f" - ML Model ROC-AUC Score : {metrics['roc_auc']}")
    print("=" * 65 + "\n")

    return {
        "bronze": bronze_summary,
        "silver": silver_summary,
        "ml_metrics": metrics,
        "elapsed_seconds": elapsed
    }

if __name__ == "__main__":
    run_full_lakehouse_pipeline(generate_fresh_data=True)
