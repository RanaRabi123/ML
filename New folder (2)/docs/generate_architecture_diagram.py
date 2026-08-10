from graphviz import Digraph

dot = Digraph("Architecture", format="png")
dot.attr(rankdir="TB", bgcolor="white", fontname="Helvetica", splines="ortho")
dot.attr("node", fontname="Helvetica", fontsize="11", shape="box", style="rounded,filled")

# --- Data layer ---
with dot.subgraph(name="cluster_data") as c:
    c.attr(label="Data Layer", style="rounded", color="#4C72B0")
    c.node("raw", "Raw CSV\n(train.csv)", fillcolor="#DCE6F1")
    c.node("etl", "ETL Pipeline\n(src/etl/clean_data.py)", fillcolor="#DCE6F1")
    c.node("processed", "Cleaned Data\n(clean_sales.csv)", fillcolor="#DCE6F1")
    c.edge("raw", "etl")
    c.edge("etl", "processed")

# --- ML layer ---
with dot.subgraph(name="cluster_ml") as c:
    c.attr(label="Modeling Layer", style="rounded", color="#55A868")
    c.node("features", "Feature Engineering\n(lags, rolling mean, encoding)", fillcolor="#DCECDA")
    c.node("train", "Model Training\n(XGBoost Regressor)", fillcolor="#DCECDA")
    c.node("model", "Trained Model\n(sales_model.joblib)", fillcolor="#DCECDA")
    c.edge("features", "train")
    c.edge("train", "model")

dot.edge("processed", "features")

# --- Serving layer ---
with dot.subgraph(name="cluster_api") as c:
    c.attr(label="Application / Serving Layer", style="rounded", color="#C44E52")
    c.node("api", "FastAPI Service\n/auth/login (JWT)\n/predict/sales\n/health", fillcolor="#F2DCDB")
    c.node("dashboard", "Streamlit Dashboard\n(BI charts + prediction form)", fillcolor="#F2DCDB")

dot.edge("model", "api", label="  loads model")
dot.edge("api", "dashboard", label="  REST calls (JWT)")
dot.edge("processed", "dashboard", label="  reads for BI charts")

# --- User layer ---
dot.node("user", "Business User / Analyst", shape="ellipse", fillcolor="#FFF2CC")
dot.edge("user", "dashboard", label="  logs in, views KPIs, requests forecast")

# --- Cross-cutting concerns ---
with dot.subgraph(name="cluster_cross") as c:
    c.attr(label="Cross-Cutting Concerns", style="dashed", color="gray40")
    c.node("logging", "Logging & Error Handling", fillcolor="#EDEDED", shape="note")
    c.node("testing", "Automated Tests (pytest)", fillcolor="#EDEDED", shape="note")
    c.node("security", "JWT Auth, Env-based Secrets", fillcolor="#EDEDED", shape="note")

dot.render("docs/architecture_diagram", cleanup=True)
print("Diagram saved to docs/architecture_diagram.png")
