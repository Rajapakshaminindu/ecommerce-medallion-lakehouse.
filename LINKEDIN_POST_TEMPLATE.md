# 📱 LinkedIn Post Template

Copy and customize this post for LinkedIn when sharing your new project:

---

🚀 **Excited to share my latest project: Building an End-to-End E-Commerce Lakehouse & Machine Learning Pipeline using the Medallion Architecture (Bronze 🥉 ➔ Silver 🥈 ➔ Gold 🥇)!**

As I continue deepening my hands-on experience in **Data Science & Cloud Data Engineering**, I wanted to move beyond simple CSV analysis and build a scalable **ELT (Extract, Load, Transform)** architecture from scratch using Python, Parquet, Scikit-Learn, and Streamlit.

### 🔍 Key Highlights of the Project:
1. **🥉 Bronze Layer (Raw Ingestion):**
   - Implemented append-only ingestion preserving exact source fidelity.
   - Attached audit metadata (`_ingested_at`, `_source_file`, `_batch_id`) in optimized columnar Parquet format.

2. **🥈 Silver Layer (Cleansed & Enriched):**
   - Built automated data quality checks: deduplication on `order_id`, anomaly filtering (handling invalid quantities/prices), and string standardizations.
   - Performed relational joins with Customer & Product dimensions to compute real-time financial margins.

3. **🥇 Gold Layer (Curated KPIs & ML Feature Store):**
   - Structured aggregated tables for executive dashboards (daily revenue rollups, category margins).
   - Engineered an **RFM (Recency, Frequency, Monetary) Customer Feature Store** tailored for machine learning.

4. **🤖 Machine Learning & Live Inference UI:**
   - Trained a **Random Forest Churn Prediction Model** (ROC-AUC > 0.90) with 5-fold cross-validation.
   - Built a full **Streamlit Web Application** featuring live pipeline monitoring, layer data exploration, and an interactive customer churn risk simulator!

💡 **Key Takeaway:** Real-world data science is deeply connected to solid data engineering. Understanding how data moves through Bronze, Silver, and Gold layers ensures that ML models consume reliable, high-quality feature stores.

🔗 **GitHub Repository:** [Insert your GitHub Repo Link Here]

I’d love to hear your thoughts and feedback! 🙌

#DataScience #MachineLearning #DataEngineering #Python #MedallionArchitecture #Lakehouse #Streamlit #ScikitLearn #ELT #PortfolioProject
