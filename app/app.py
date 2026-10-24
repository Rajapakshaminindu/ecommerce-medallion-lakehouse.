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
from src.ml_model import CustomerChurnMLModel, CustomerLTVMLModel
from src.predict_batch import BatchScoringEngine

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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏗️ Pipeline Architecture & Flow",
    "🔍 Lakehouse Layer Explorer",
    "📊 Executive KPI Analytics",
    "🤖 Live Machine Learning Predictor",
    "🎯 Retention Action Center & CRM Export"
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
        table_pick = st.selectbox("Select Gold Table:", [
            "gold_kpi_daily_revenue.parquet",
            "gold_kpi_category_performance.parquet",
            "gold_customer_ml_feature_store.parquet",
            "gold_cohort_retention.parquet",
            "gold_customer_churn_predictions.parquet"
        ])
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

    # Cohort Retention Matrix Heatmap
    st.markdown("---")
    st.subheader("📅 Monthly Customer Cohort Retention Analysis")
    cohort_path = os.path.join(GOLD_DIR, "gold_cohort_retention.parquet")
    if os.path.exists(cohort_path):
        df_cohort = pd.read_parquet(cohort_path)
        cohort_pivot = df_cohort.pivot(index="cohort_month", columns="period_number", values="retention_rate_pct")
        cohort_pivot = cohort_pivot.sort_index(ascending=False)
        col_names = [f"Month {int(c)}" for c in cohort_pivot.columns]

        fig_cohort = px.imshow(
            cohort_pivot.values,
            x=col_names,
            y=[str(idx) for idx in cohort_pivot.index],
            labels=dict(x="Time Since Acquisition", y="Acquisition Cohort", color="Retention (%)"),
            color_continuous_scale="Viridis",
            text_auto=".1f",
            aspect="auto"
        )
        fig_cohort.update_layout(
            title="Monthly Customer Cohort Retention Heatmap (%)",
            xaxis_title="Cohort Age (Months Since First Order)",
            yaxis_title="Acquisition Month",
            height=340
        )
        st.plotly_chart(fig_cohort, use_container_width=True)

        m1_vals = df_cohort[df_cohort["period_number"] == 1]["retention_rate_pct"]
        m2_vals = df_cohort[df_cohort["period_number"] == 2]["retention_rate_pct"]
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Total Tracked Cohorts", f"{df_cohort['cohort_month'].nunique()} cohorts")
        with s2:
            st.metric("Avg Month 1 Retention", f"{m1_vals.mean():.1f}%" if len(m1_vals) > 0 else "N/A")
        with s3:
            st.metric("Avg Month 2 Retention", f"{m2_vals.mean():.1f}%" if len(m2_vals) > 0 else "N/A")

    # Customer Lifetime Value (LTV) Analytics
    st.markdown("---")
    st.subheader("💰 Customer Lifetime Value (LTV) Distribution")
    ltv_c1, ltv_c2 = st.columns(2)
    with ltv_c1:
        avg_ltv_tier = df_features.groupby("loyalty_tier")["total_monetary_spend"].mean().reset_index()
        fig_ltv_tier = px.bar(
            avg_ltv_tier, x="loyalty_tier", y="total_monetary_spend",
            title="Average Lifetime Value by Loyalty Tier ($)",
            labels={"loyalty_tier": "Loyalty Tier", "total_monetary_spend": "Average Spend ($)"},
            color="total_monetary_spend", color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_ltv_tier, use_container_width=True)
    with ltv_c2:
        fig_ltv_hist = px.histogram(
            df_features, x="total_monetary_spend", nbins=20,
            title="Customer Lifetime Value Distribution ($)",
            labels={"total_monetary_spend": "Customer Lifetime Value ($)"},
            color_discrete_sequence=["#7E57C2"]
        )
        st.plotly_chart(fig_ltv_hist, use_container_width=True)

# -------------------------------------------------------------
# TAB 4: LIVE MACHINE LEARNING PREDICTOR
# -------------------------------------------------------------
with tab4:
    st.subheader("Customer Churn Risk & Lifetime Value (LTV) Intelligence")

    # Load ML Metadata for Churn and LTV
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    ltv_meta_path = os.path.join(MODELS_DIR, "ltv_model_metadata.json")

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            ml_meta = json.load(f)

        st.markdown("#### 🎯 Churn Classifier Metrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("ROC-AUC Score", f"{ml_meta['metrics']['roc_auc']:.4f}")
        m2.metric("5-Fold CV AUC", f"{ml_meta['metrics']['cv_roc_auc_mean']:.4f}")
        m3.metric("Precision", f"{ml_meta['metrics']['precision']:.4f}")
        m4.metric("Recall", f"{ml_meta['metrics']['recall']:.4f}")
        m5.metric("F1-Score", f"{ml_meta['metrics']['f1_score']:.4f}")

        if os.path.exists(ltv_meta_path):
            with open(ltv_meta_path, "r") as f:
                ltv_meta = json.load(f)
            st.markdown("#### 📈 Lifetime Value (LTV) Regressor Metrics")
            lm1, lm2, lm3, lm4 = st.columns(4)
            lm1.metric("LTV R² Score", f"{ltv_meta['metrics']['r2_score']:.4f}")
            lm2.metric("Mean Absolute Error (MAE)", f"${ltv_meta['metrics']['mae']:,.2f}")
            lm3.metric("Root Mean Squared Error", f"${ltv_meta['metrics']['rmse']:,.2f}")
            lm4.metric("5-Fold CV R²", f"{ltv_meta['metrics']['cv_r2_mean']:.4f}")

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
            st.markdown("### 🎛️ Live Customer Risk & LTV Simulator")
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
                
                sim_submit = st.form_submit_button("🔮 Predict Customer Churn & Lifetime Value", type="primary", use_container_width=True)

            if sim_submit:
                churn_engine = CustomerChurnMLModel(GOLD_DIR, MODELS_DIR)
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
                pred_result = churn_engine.predict_single_customer(input_payload)
                churn_prob = pred_result["churn_risk_probability"]

                ltv_engine = CustomerLTVMLModel(GOLD_DIR, MODELS_DIR)
                pred_ltv = ltv_engine.predict_single_customer_ltv(input_payload)

                res_c1, res_c2 = st.columns(2)
                with res_c1:
                    if churn_prob >= 0.60:
                        st.error(f"🚨 **High Churn Risk!** Probability: **{churn_prob*100:.1f}%**")
                        st.warning("💡 **Actionable Recommendation:** Send an automated 15% win-back discount or customer care follow-up.")
                    elif churn_prob >= 0.35:
                        st.warning(f"⚠️ **Moderate Churn Risk.** Probability: **{churn_prob*100:.1f}%**")
                        st.info("💡 **Actionable Recommendation:** Enroll into loyalty points acceleration program.")
                    else:
                        st.success(f"✅ **Low Churn Risk / Highly Engaged!** Probability: **{churn_prob*100:.1f}%**")
                        st.info("💡 **Actionable Recommendation:** Eligible for premium upsell campaigns.")

                with res_c2:
                    st.metric("💎 Predicted Customer Lifetime Value (LTV)", f"${pred_ltv['predicted_ltv']:,.2f}")
    else:
        st.warning("Train the machine learning models first by running the pipeline.")

# -------------------------------------------------------------
# TAB 5: RETENTION ACTION CENTER & CRM EXPORT
# -------------------------------------------------------------
with tab5:
    st.subheader("🎯 Customer Retention Prioritization & Revenue-at-Risk Engine")
    st.markdown("Automated batch inference across the Gold Feature Store ranking high-value accounts facing churn risk.")

    predictions_path = os.path.join(GOLD_DIR, "gold_customer_churn_predictions.parquet")
    if not os.path.exists(predictions_path):
        with st.spinner("⚡ Running first-time Batch Scoring Engine..."):
            scoring_engine = BatchScoringEngine(gold_dir=GOLD_DIR, models_dir=MODELS_DIR)
            df_scored = scoring_engine.run_batch_scoring()
    else:
        df_scored = pd.read_parquet(predictions_path)

    # Top KPI Metrics
    crit_count = int((df_scored["risk_tier"] == "Critical Risk").sum())
    mod_count = int((df_scored["risk_tier"] == "Moderate Risk").sum())
    low_count = int((df_scored["risk_tier"] == "Low Risk").sum())
    total_rev_risk = float(df_scored["revenue_at_risk"].sum())
    vip_interventions = int(((df_scored["risk_tier"] == "Critical Risk") & (df_scored["predicted_ltv"] >= 500)).sum())

    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        st.metric("Total Scored Customers", f"{len(df_scored):,}", delta="Gold ML Store")
    with rc2:
        st.metric("Critical Churn Risk (≥60%)", f"{crit_count:,}", delta=f"{round(crit_count/len(df_scored)*100, 1)}% of base", delta_color="inverse")
    with rc3:
        st.metric("Total Revenue at Risk", f"${total_rev_risk:,.2f}", delta="Expected Lost Value", delta_color="inverse")
    with rc4:
        st.metric("VIP Priority Interventions", f"{vip_interventions:,}", delta="High LTV & Churn", delta_color="inverse")

    st.markdown("---")

    # Visual Analytics
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        rev_by_tier = df_scored.groupby("loyalty_tier")["revenue_at_risk"].sum().reset_index()
        fig_rev_tier = px.bar(
            rev_by_tier, x="loyalty_tier", y="revenue_at_risk",
            title="Total Revenue at Risk by Loyalty Tier ($)",
            labels={"loyalty_tier": "Loyalty Tier", "revenue_at_risk": "Revenue at Risk ($)"},
            color="revenue_at_risk", color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_rev_tier, use_container_width=True)

    with r_col2:
        tier_counts = pd.DataFrame({
            "Risk Tier": ["Critical Risk", "Moderate Risk", "Low Risk"],
            "Count": [crit_count, mod_count, low_count]
        })
        fig_risk_pie = px.pie(
            tier_counts, names="Risk Tier", values="Count",
            title="Customer Base Risk Segmentation",
            hole=0.45,
            color="Risk Tier",
            color_discrete_map={"Critical Risk": "#D32F2F", "Moderate Risk": "#FFA000", "Low Risk": "#2E7D32"}
        )
        st.plotly_chart(fig_risk_pie, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Targeted Retention Intervention List")

    # Interactive Table Filters
    f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
    with f_col1:
        risk_options = ["All"] + list(df_scored["risk_tier"].unique())
        selected_risk = st.selectbox("Filter by Risk Tier:", risk_options, index=0)
    with f_col2:
        tier_options = ["All"] + list(df_scored["loyalty_tier"].unique())
        selected_tier = st.selectbox("Filter by Loyalty Tier:", tier_options, index=0)
    with f_col3:
        search_query = st.text_input("🔍 Search Customer ID:", placeholder="e.g. CUST_0016")

    df_filtered = df_scored.copy()
    if selected_risk != "All":
        df_filtered = df_filtered[df_filtered["risk_tier"] == selected_risk]
    if selected_tier != "All":
        df_filtered = df_filtered[df_filtered["loyalty_tier"] == selected_tier]
    if search_query.strip():
        df_filtered = df_filtered[df_filtered["customer_id"].str.contains(search_query.strip(), case=False, na=False)]

    display_cols = [
        "customer_id", "loyalty_tier", "region", "total_monetary_spend",
        "recency_days", "churn_risk_probability", "predicted_ltv",
        "revenue_at_risk", "risk_tier", "recommended_action"
    ]
    st.dataframe(df_filtered[display_cols], use_container_width=True)

    # Download Button for CRM Campaign Export
    csv_bytes = df_filtered[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Retention Action List (CSV)",
        data=csv_bytes,
        file_name="customer_retention_action_campaign.csv",
        mime="text/csv",
        type="primary"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9rem;'>
    Built with Python, Pandas, Scikit-Learn, Parquet & Streamlit | Medallion Lakehouse Architecture
</div>
""", unsafe_allow_html=True)
