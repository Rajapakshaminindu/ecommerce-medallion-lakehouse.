"""
Bronze Layer Pipeline Engine
Responsible for raw ingestion (ELT), preserving exact source fidelity,
and attaching audit metadata (_ingested_at, _source_file, _batch_id).
Stores data in columnar Parquet format.
"""

import os
import glob
from datetime import datetime, timezone
import pandas as pd

class BronzePipeline:
    def __init__(self, raw_dir: str, bronze_dir: str):
        self.raw_dir = raw_dir
        self.bronze_dir = bronze_dir
        os.makedirs(self.bronze_dir, exist_ok=True)

    def ingest_master_tables(self):
        """Ingests static / dimension master files (Customers, Products) into Bronze."""
        ingestion_results = {}
        for table_name in ["customers", "products"]:
            src_file = os.path.join(self.raw_dir, f"{table_name}.csv")
            if not os.path.exists(src_file):
                print(f"[WARN] Source file missing: {src_file}")
                continue

            df = pd.read_csv(src_file)
            # Add Bronze Audit Metadata
            df["_ingested_at"] = datetime.now(timezone.utc).isoformat()
            df["_source_file"] = os.path.basename(src_file)
            df["_bronze_version"] = "v1"

            target_path = os.path.join(self.bronze_dir, f"bronze_{table_name}.parquet")
            df.to_parquet(target_path, index=False)
            ingestion_results[table_name] = len(df)
            print(f"[Bronze Ingested] {table_name}: {len(df)} rows -> {target_path}")

        return ingestion_results

    def ingest_order_batches(self):
        """Ingests all raw order batches into an append-only Bronze Orders table."""
        batch_files = sorted(glob.glob(os.path.join(self.raw_dir, "orders_batch_*.csv")))
        if not batch_files:
            print("[WARN] No order batches found to ingest.")
            return {"orders": 0}

        all_bronze_orders = []
        for file_path in batch_files:
            batch_name = os.path.basename(file_path)
            df_batch = pd.read_csv(file_path)
            
            # Attach Bronze Ingestion Metadata
            df_batch["_ingested_at"] = datetime.now(timezone.utc).isoformat()
            df_batch["_source_file"] = batch_name
            df_batch["_batch_id"] = batch_name.replace(".csv", "")
            
            all_bronze_orders.append(df_batch)

        df_consolidated = pd.concat(all_bronze_orders, ignore_index=True)
        target_path = os.path.join(self.bronze_dir, "bronze_orders.parquet")
        df_consolidated.to_parquet(target_path, index=False)
        print(f"[Bronze Ingested] Raw Orders ({len(batch_files)} batches): {len(df_consolidated)} total records -> {target_path}")

        return {"orders": len(df_consolidated), "batches": len(batch_files)}

    def run(self):
        print("\n==========================================")
        print(">>> RUNNING BRONZE INGESTION LAYER")
        print("==========================================")
        masters = self.ingest_master_tables()
        orders = self.ingest_order_batches()
        print("[SUCCESS] Bronze Layer Ingestion Complete.\n")
        return {"masters": masters, "orders": orders}

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    raw_path = os.path.join(base_dir, "data", "raw_source")
    bronze_path = os.path.join(base_dir, "data", "lakehouse", "bronze")
    
    pipeline = BronzePipeline(raw_path, bronze_path)
    pipeline.run()
