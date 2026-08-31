"""
Interactive Streamlit Application
E-Commerce Medallion Lakehouse & Machine Learning System
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="E-Commerce Medallion Lakehouse & ML",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

RAW_DIR = os.path.join(BASE_DIR, "data", "raw_source")
BRONZE_DIR = os.path.join(BASE_DIR, "data", "lakehouse", "bronze")
SILVER_DIR = os.path.join(BASE_DIR, "data", "lakehouse", "silver")
GOLD_DIR = os.path.join(BASE_DIR, "data", "lakehouse", "gold")
MODELS_DIR = os.path.join(BASE_DIR, "models")

from src.run_pipeline import run_full_lakehouse_pipeline
from src.ml_model import CustomerChurnMLModel

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5, #7E57C2, #43A047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .badge-bronze { background-color: #CD7F32; color: white; padding: 3px 8px; border-radius: 5px; font-weight: bold; }
    .badge-silver { background-color: #9E9E9E; color: white; padding: 3px 8px; border-radius: 5px; font-weight: bold; }
    .badge-gold { background-color: #FFD700; color: #212529; padding: 3px 8px; border-radius: 5px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Helper function to check if lakehouse data exists
def is_lakehouse_ready():
    bronze_file = os.path.join(BRONZE_DIR, "bronze_orders.parquet")
    silver_file = os.path.join(SILVER_DIR, "silver_orders_enriched.parquet")
    gold_file = os.path.join(GOLD_DIR, "gold_customer_ml_feature_store.parquet")
    return os.path.exists(bronze_file) and os.path.exists(silver_file) and os.path.exists(gold_file)

# Ensure data is initialized on first load
if not is_lakehouse_ready():
    with st.spinner("🚀 Initializing Lakehouse & running first-time ELT pipeline..."):
        run_full_lakehouse_pipeline(generate_fresh_data=True)

# Sidebar
with st.sidebar:
    st.image("https://img.shields.io/badge/Architecture-Medallion%20ELT-blue?style=for-the-badge&logo=apachespark", use_container_width=True)
    st.title("⚙️ Pipeline Controls")
    
    st.markdown("---")
    if st.button("🔄 Run Full ELT Pipeline", type="primary", use_container_width=True):
        with st.spinner("Executing Bronze ➔ Silver ➔ Gold ➔ ML Training..."):
            summary = run_full_lakehouse_pipeline(generate_fresh_data=True)
            st.success(f"Pipeline executed in {summary['elapsed_seconds']}s!")
            st.rerun()

    st.markdown("---")
    st.markdown("### 📚 Architecture Guide")
    st.markdown("🥉 **Bronze**: Raw ingest + audit timestamps")
    st.markdown("🥈 **Silver**: Cleaning, typing, deduplication")
    st.markdown("🥇 **Gold**: Business KPIs & ML Feature Store")
    st.markdown("🤖 **ML**: Churn risk classifier + inference")

# Main Header
st.markdown('<div class="main-header">⚡ E-Commerce Medallion Lakehouse & AI System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production-ready ELT Data Pipeline (Bronze ➔ Silver ➔ Gold) with Integrated Machine Learning</div>', unsafe_allow_html=True)

# Top KPI Summary Cards
try:
    df_silver_orders = pd.read_parquet(os.path.join(SILVER_DIR, "silver_orders_enriched.parquet"))
    df_bronze_orders = pd.read_parquet(os.path.join(BRONZE_DIR, "bronze_orders.parquet"))
    df_gold_kpi = pd.read_parquet(os.path.join(GOLD_DIR, "gold_kpi_daily_revenue.parquet"))
    df_features = pd.read_parquet(os.path.join(GOLD_DIR, "gold_customer_ml_feature_store.parquet"))

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Raw Ingested (Bronze)", f"{len(df_bronze_orders):,}", delta="Raw Events")
    with c2:
        st.metric("Clean Orders (Silver)", f"{len(df_silver_orders):,}", delta=f"{round((len(df_silver_orders)/len(df_bronze_orders))*100, 1)}% Clean")
    with c3:
        st.metric("Total Net Revenue", f"${df_silver_orders['net_revenue'].sum():,.2f}")
    with c4:
        st.metric("Total Net Profit", f"${df_silver_orders['profit'].sum():,.2f}", delta=f"{round((df_silver_orders['profit'].sum()/df_silver_orders['net_revenue'].sum())*100, 1)}% Margin")
    with c5:
        st.metric("Active Customers", f"{len(df_features):,}", delta="Feature Vectors")
except Exception as e:
    st.warning(f"Run the pipeline to populate lakehouse metrics: {e}")

st.markdown("---")

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🏗️ Pipeline Architecture & Flow",
    "🔍 Lakehouse Layer Explorer",
    "📊 Executive KPI Analytics",
    "🤖 Live Machine Learning Predictor"
])

# -------------------------------------------------------------
# TAB 1: ARCHITECTURE & PIPELINE FLOW
# -------------------------------------------------------------
with tab1:
    st.subheader("Data Flow Lifecycle (Medallion Pattern)")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("### 🥉 Bronze Layer (Raw)")
        st.info("""
        - **Source:** Batch CSV / JSON feeds
        - **Operation:** Append-only ingestion
        - **Schema:** Source-exact, zero data loss
        - **Audit:** `_ingested_at`, `_source_file`, `_batch_id`
        - **Format:** Optimized Columnar Parquet
        """)
        st.metric("Bronze Records", f"{len(df_bronze_orders)} rows")
        
    with col_b:
        st.markdown("### 🥈 Silver Layer (Cleaned)")
        st.info("""
        - **Operation:** Type coercion & standard formatting
        - **Deduplication:** Unique order ID enforcement
        - **Validation:** Quantity > 0, Price > 0 filtering
        - **Enrichment:** Relational joins (Customers + Products)
        - **Calculations:** Gross, Net Revenue, Cost, Profit Margin
        """)
        st.metric("Silver Records", f"{len(df_silver_orders)} valid rows")

    with col_c:
        st.markdown("### 🥇 Gold Layer (Curated & ML)")
        st.info("""
        - **Operation:** Business logic aggregations
        - **Analytics:** Daily KPI rollups, Category performance
        - **Feature Store:** RFM (Recency, Frequency, Monetary)
        - **Consumers:** Power BI, Tableau, ML Churn Model
        - **Targets:** `is_churn_risk`, `is_high_value`
        """)
        st.metric("Gold Feature Vectors", f"{len(df_features)} customers")

    st.markdown("---")
    st.subheader("🛡️ Automated Data Quality & Anomaly Detection Report")
    
    dup_count = len(df_bronze_orders) - df_bronze_orders["order_id"].nunique()
    invalid_qty = len(df_bronze_orders[df_bronze_orders["quantity"] <= 0]) if "quantity" in df_bronze_orders.columns else 0
    clean_pass = len(df_silver_orders)
    
    dq_col1, dq_col2 = st.columns([1, 2])
    with dq_col1:
        st.markdown(f"""
        - **Initial Bronze Records:** `{len(df_bronze_orders)}`
        - **Duplicates Identified & Purged:** `{dup_count}`
        - **Corrupted Records Filtered:** `{invalid_qty}`
        - **Final Silver Clean Records:** `{clean_pass}`
        - **Overall Quality Pass Rate:** `{(clean_pass/len(df_bronze_orders))*100:.2f}%`
        """)
    with dq_col2:
        dq_df = pd.DataFrame({
            "Stage": ["Valid Clean Orders", "Duplicate Orders Removed", "Corrupted Records Filtered"],
            "Count": [clean_pass, dup_count, invalid_qty]
        })
        fig_dq = px.pie(dq_df, values="Count", names="Stage", color_discrete_sequence=["#2E7D32", "#FFA000", "#D32F2F"], hole=0.4)
        fig_dq.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=220)
        st.plotly_chart(fig_dq, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: LAKEHOUSE LAYER EXPLORER
# -------------------------------------------------------------
with tab2:
    st.subheader("Explore Tables Across Lakehouse Zones")
    
    layer_selection = st.radio(
        "Select Lakehouse Layer to Inspect:",
        ["🥉 Bronze (Raw Ingest)", "🥈 Silver (Cleansed & Enriched)", "🥇 Gold (Business KPIs & Feature Store)"],
        horizontal=True
    )

    if "Bronze" in layer_selection:
        table_pick = st.selectbox("Select Bronze Table:", ["bronze_orders.parquet", "bronze_customers.parquet", "bronze_products.parquet"])
        df_view = pd.read_parquet(os.path.join(BRONZE_DIR, table_pick))
        st.markdown(f"**Viewing `{table_pick}`** ({len(df_view)} rows, {len(df_view.columns)} columns)")
        st.dataframe(df_view.head(50), use_container_width=True)

    elif "Silver" in layer_selection:
        table_pick = st.selectbox("Select Silver Table:", ["silver_orders_enriched.parquet", "silver_customers.parquet", "silver_products.parquet"])
        df_view = pd.read_parquet(os.path.join(SILVER_DIR, table_pick))
        st.markdown(f"**Viewing `{table_pick}`** ({len(df_view)} rows, {len(df_view.columns)} columns)")
        st.dataframe(df_view.head(50), use_container_width=True)

    else:
        table_pick = st.selectbox("Select Gold Table:", ["gold_kpi_daily_revenue.parquet", "gold_kpi_category_performance.parquet", "gold_customer_ml_feature_store.parquet"])
        df_view = pd.read_parquet(os.path.join(GOLD_DIR, table_pick))
        st.markdown(f"**Viewing `{table_pick}`** ({len(df_view)} rows, {len(df_view.columns)} columns)")
        st.dataframe(df_view.head(50), use_container_width=True)

# -------------------------------------------------------------
# TAB 3: EXECUTIVE KPI ANALYTICS
# -------------------------------------------------------------
with tab3:
    st.subheader("Executive Business Performance Dashboard")
    
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig_rev = px.line(
            df_gold_kpi, x="order_date_day", y=["net_revenue", "total_profit"],
            title="Daily Net Revenue & Profit Trends ($)",
            labels={"value": "Amount ($)", "order_date_day": "Date", "variable": "Metric"},
            color_discrete_map={"net_revenue": "#1E88E5", "total_profit": "#43A047"}
        )
        fig_rev.update_layout(hovermode="x unified")
        st.plotly_chart(fig_rev, use_container_width=True)

    with chart_col2:
        df_cat_kpi = pd.read_parquet(os.path.join(GOLD_DIR, "gold_kpi_category_performance.parquet"))
        fig_cat = px.bar(
            df_cat_kpi, x="category", y="net_revenue", color="profit_margin_pct",
            title="Category Revenue & Profit Margin (%)",
            color_continuous_scale="Blues",
            labels={"net_revenue": "Net Revenue ($)", "profit_margin_pct": "Margin %"}
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        fig_reg = px.pie(
            df_silver_orders, names="region", values="net_revenue",
            title="Revenue Contribution by Geographic Region",
            hole=0.4, color_discrete_sequence=px.colors.sequential.Teal
        )
        st.plotly_chart(fig_reg, use_container_width=True)

    with chart_col4:
        fig_scatter = px.scatter(
            df_features, x="recency_days", y="total_monetary_spend", color="loyalty_tier",
            size="frequency", hover_data=["customer_id", "avg_order_value"],
            title="Customer RFM Distribution (Spend vs Recency)",
            labels={"recency_days": "Recency (Days Inactive)", "total_monetary_spend": "Total Spend ($)"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# -------------------------------------------------------------
# TAB 4: LIVE MACHINE LEARNING PREDICTOR
# -------------------------------------------------------------
with tab4:
    st.subheader("Customer Churn Risk & Retention Intelligence")

    # Load ML Metadata
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            ml_meta = json.load(f)
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("ROC-AUC Score", f"{ml_meta['metrics']['roc_auc']:.4f}")
        m2.metric("5-Fold CV AUC", f"{ml_meta['metrics']['cv_roc_auc_mean']:.4f}")
        m3.metric("Precision", f"{ml_meta['metrics']['precision']:.4f}")
        m4.metric("Recall", f"{ml_meta['metrics']['recall']:.4f}")
        m5.metric("F1-Score", f"{ml_meta['metrics']['f1_score']:.4f}")

        st.markdown("---")
        
        ml_col1, ml_col2 = st.columns([1, 1])
        
        with ml_col1:
            st.markdown("### 🔍 Model Feature Importance")
            df_imp = pd.DataFrame(ml_meta["feature_importance"]).head(8)
            fig_imp = px.bar(
                df_imp, x="importance", y="feature", orientation="h",
                title="Top Predictive Features (Random Forest)",
                color="importance", color_continuous_scale="Purples"
            )
            fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=320)
            st.plotly_chart(fig_imp, use_container_width=True)

        with ml_col2:
            st.markdown("### 🎛️ Live Customer Risk Simulator")
            with st.form("churn_simulator_form"):
                fc1, fc2 = st.columns(2)
                with fc1:
                    sim_recency = st.slider("Recency (Days since last purchase)", 1, 365, 140)
                    sim_freq = st.slider("Order Frequency (Total orders)", 1, 30, 2)
                    sim_spend = st.number_input("Total Monetary Spend ($)", 10.0, 5000.0, 180.0)
                    sim_aov = st.number_input("Average Order Value ($)", 10.0, 1000.0, 90.0)
                with fc2:
                    sim_return_rate = st.slider("Return Rate (%)", 0.0, 100.0, 25.0)
                    sim_age = st.slider("Customer Age", 18, 75, 32)
                    sim_tier = st.selectbox("Loyalty Tier", ["Regular", "Silver", "Gold", "Platinum"])
                    sim_region = st.selectbox("Region", ["Western", "Central", "Southern", "North Western", "Northern"])
                
                sim_submit = st.form_submit_button("🔮 Predict Customer Churn Risk", type="primary", use_container_width=True)

            if sim_submit:
                ml_engine = CustomerChurnMLModel(GOLD_DIR, MODELS_DIR)
                input_payload = {
                    "frequency": sim_freq,
                    "total_monetary_spend": sim_spend,
                    "avg_order_value": sim_aov,
                    "total_units_purchased": sim_freq * 2,
                    "return_rate_pct": sim_return_rate,
                    "recency_days": sim_recency,
                    "customer_tenure_days": sim_recency + 60,
                    "customer_age": sim_age,
                    "region": sim_region,
                    "loyalty_tier": sim_tier
                }
                pred_result = ml_engine.predict_single_customer(input_payload)
                churn_prob = pred_result["churn_risk_probability"]

                if churn_prob >= 0.60:
                    st.error(f"🚨 **High Churn Risk!** Probability: **{churn_prob*100:.1f}%**")
                    st.warning("💡 **Actionable Recommendation:** Send an automated 15% win-back discount or customer care follow-up.")
                elif churn_prob >= 0.35:
                    st.warning(f"⚠️ **Moderate Churn Risk.** Probability: **{churn_prob*100:.1f}%**")
                    st.info("💡 **Actionable Recommendation:** Enroll into loyalty points acceleration program.")
                else:
                    st.success(f"✅ **Low Churn Risk / Highly Engaged!** Probability: **{churn_prob*100:.1f}%**")
                    st.info("💡 **Actionable Recommendation:** Eligible for premium upsell campaigns.")
    else:
        st.warning("Train the machine learning model first by running the pipeline.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9rem;'>
    Built with Python, Pandas, Scikit-Learn, Parquet & Streamlit | Medallion Lakehouse Architecture
</div>
""", unsafe_allow_html=True)
