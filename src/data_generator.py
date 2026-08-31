"""
Synthetic E-Commerce Data Generator
Simulates realistic raw transaction batches with deliberate data quality anomalies
to demonstrate robust ELT data cleaning in the Silver Layer.
"""

import os
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def generate_synthetic_dataset(output_dir: str, num_customers: int = 250, num_orders: int = 1500, seed: int = 42):
    """
    Generates customers, products, and raw batch order files with realistic messy data.
    """
    np.random.seed(seed)
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Generate Products Master Data
    categories = {
        "Electronics": [("Wireless Headphones", 89.99, 45.00), ("Smart Watch", 199.99, 110.00), 
                        ("Mechanical Keyboard", 129.50, 60.00), ("USB-C Dock", 65.00, 30.00)],
        "Fashion": [("Cotton Hoodie", 49.99, 18.00), ("Running Shoes", 110.00, 50.00), 
                    ("Denim Jeans", 69.90, 25.00), ("Leather Belt", 35.00, 12.00)],
        "Home & Kitchen": [("Espresso Machine", 249.99, 130.00), ("Blender", 79.99, 35.00), 
                           ("Non-Stick Cookware Set", 145.00, 70.00), ("Air Purifier", 160.00, 85.00)],
        "Fitness": [("Yoga Mat", 29.99, 10.00), ("Adjustable Dumbbell Set", 189.00, 95.00), 
                    ("Resistance Bands", 19.99, 6.00), ("Foam Roller", 24.50, 8.00)]
    }

    products_data = []
    prod_id = 1
    for cat, items in categories.items():
        for name, price, cost in items:
            products_data.append({
                "product_id": f"PRD_{prod_id:03d}",
                "product_name": name,
                "category": cat,
                "unit_price": price,
                "unit_cost": cost
            })
            prod_id += 1

    df_products = pd.DataFrame(products_data)
    df_products.to_csv(os.path.join(output_dir, "products.csv"), index=False)

    # 2. Generate Customers Master Data
    regions = ["Western", "Central", "Southern", "North Western", "Northern"]
    loyalty_tiers = ["Regular", "Silver", "Gold", "Platinum"]
    
    customers_data = []
    base_date = datetime(2025, 1, 1)
    for i in range(1, num_customers + 1):
        signup_offset = random.randint(0, 365)
        signup_dt = base_date + timedelta(days=signup_offset)
        customers_data.append({
            "customer_id": f"CUST_{i:04d}",
            "customer_name": f"Customer_{i}",
            "region": random.choice(regions),
            "loyalty_tier": random.choices(loyalty_tiers, weights=[0.55, 0.25, 0.15, 0.05])[0],
            "signup_date": signup_dt.strftime("%Y-%m-%d"),
            "customer_age": random.randint(18, 65)
        })

    df_customers = pd.DataFrame(customers_data)
    df_customers.to_csv(os.path.join(output_dir, "customers.csv"), index=False)

    # 3. Generate Raw Orders in Batches (simulating continuous ingestion batches)
    # Split orders into 2 batches
    batch_size = num_orders // 2
    order_id_counter = 10001
    
    for batch_num in [1, 2]:
        orders_data = []
        for _ in range(batch_size):
            cust = random.choice(customers_data)
            prod = random.choice(products_data)
            order_days = random.randint(30, 450)
            order_dt = base_date + timedelta(days=order_days)
            quantity = random.choices([1, 2, 3, 4, 5], weights=[0.60, 0.25, 0.08, 0.04, 0.03])[0]
            status = random.choices(["COMPLETED", "SHIPPED", "RETURNED", "CANCELLED"], weights=[0.75, 0.12, 0.08, 0.05])[0]
            discount_pct = random.choices([0.0, 0.05, 0.10, 0.15, 0.20], weights=[0.6, 0.15, 0.1, 0.1, 0.05])[0]
            
            raw_order = {
                "order_id": f"ORD_{order_id_counter}",
                "customer_id": cust["customer_id"],
                "product_id": prod["product_id"],
                "order_date": order_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "quantity": quantity,
                "unit_price": prod["unit_price"],
                "discount_pct": discount_pct,
                "order_status": status,
                "payment_method": random.choice(["Credit Card", "Debit Card", "Bank Transfer", "Digital Wallet"])
            }
            orders_data.append(raw_order)
            order_id_counter += 1

        df_orders = pd.DataFrame(orders_data)

        # Inject realistic data quality anomalies (to test Silver layer cleaning)
        # 1. Add duplicates
        dup_indices = random.sample(range(len(df_orders)), 15)
        duplicates = df_orders.iloc[dup_indices].copy()
        df_orders = pd.concat([df_orders, duplicates], ignore_index=True)

        # 2. Inject missing values (nulls)
        null_indices = random.sample(range(len(df_orders)), 10)
        df_orders.loc[null_indices, "payment_method"] = None

        # 3. Inject dirty strings / trailing spaces
        space_indices = random.sample(range(len(df_orders)), 20)
        df_orders.loc[space_indices, "order_status"] = df_orders.loc[space_indices, "order_status"].apply(lambda s: f"  {s}  " if s else s)

        # 4. Inject small number of corrupted numeric values (negative quantity)
        corrupted_indices = random.sample(range(len(df_orders)), 5)
        df_orders.loc[corrupted_indices, "quantity"] = -1

        df_orders.to_csv(os.path.join(output_dir, f"orders_batch_{batch_num}.csv"), index=False)

    print(f"[SUCCESS] Generated {num_customers} customers, {len(products_data)} products, and 2 raw order batches in: {output_dir}")

if __name__ == "__main__":
    target_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw_source")
    generate_synthetic_dataset(target_path)
