import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Retail Intelligence",
    page_icon="🛒",
    layout="wide"
)

# ---------------- UI STYLE ----------------

st.markdown("""
<style>

.main {
    background-color:#0e1117;
}

.block-container {
    padding-top:2rem;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    return pd.read_csv("K_class_sales_clean.csv")

df = load_data()


# ---------------- MODEL TRAINING ----------------

features = [
    "Item_MRP",
    "Item_Weight",
    "Item_Visibility",
    "Profit"
]

X = df[features]
y = df["Item_Outlet_Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)


# ---------------- SIDEBAR ----------------

st.sidebar.title("🛒 AI Retail Intelligence")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Demand Prediction",
        "Demand Forecast",
        "Inventory AI",
        "Price Optimizer",
        "Model Analytics"
    ]
)


# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    st.title("Retail Business Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Products", len(df))
    col2.metric("Average Sales", f"{df['Item_Outlet_Sales'].mean():.0f}")
    col3.metric("Average Profit %", f"{df['Profit'].mean():.1f}")

    st.markdown("---")

    sales_by_type = df.groupby("Item_Type")["Item_Outlet_Sales"].mean()

    fig = px.bar(
        sales_by_type,
        title="Average Sales by Item Type"
    )

    st.plotly_chart(fig, width="stretch")

    outlet_sales = df.groupby("Outlet_Type")["Item_Outlet_Sales"].mean()

    fig2 = px.pie(
        values=outlet_sales.values,
        names=outlet_sales.index,
        title="Sales by Outlet Type"
    )

    st.plotly_chart(fig2, width="stretch")


# ---------------- DEMAND PREDICTION ----------------

elif page == "Demand Prediction":

    st.title("Demand Prediction")

    col1, col2 = st.columns(2)

    with col1:

        item_weight = st.slider("Item Weight", 1.0, 25.0, 10.0)

        visibility = st.slider(
            "Item Visibility",
            0.0,
            0.3,
            0.05
        )

    with col2:

        mrp = st.slider("Item Price", 30, 300, 120)

        profit = st.slider("Profit Margin %", 5, 25, 12)

    if st.button("Predict Demand"):

        input_df = pd.DataFrame({

            "Item_MRP":[mrp],
            "Item_Weight":[item_weight],
            "Item_Visibility":[visibility],
            "Profit":[profit]

        })

        prediction = model.predict(input_df)[0]

        st.success(f"Predicted Demand: {prediction:.0f} units")


# ---------------- DEMAND FORECAST ----------------

elif page == "Demand Forecast":

    st.title("Demand Forecast vs Price")

    prices = np.linspace(50,300,50)

    forecast_df = pd.DataFrame({

        "Item_MRP":prices,
        "Item_Weight":[10]*50,
        "Item_Visibility":[0.05]*50,
        "Profit":[12]*50

    })

    predictions = model.predict(forecast_df)

    fig = px.line(

        x=prices,
        y=predictions,
        labels={"x":"Price","y":"Predicted Demand"},
        title="Demand Forecast vs Price"

    )

    st.plotly_chart(fig, width="stretch")


# ---------------- INVENTORY AI ----------------

elif page == "Inventory AI":

    st.title("Smart Inventory Reorder")

    item = st.selectbox(

        "Select Item Type",
        df["Item_Type"].unique()

    )

    current_stock = st.slider(

        "Current Stock",
        10,
        500,
        120

    )

    avg_sales = df[df["Item_Type"] == item]["Item_Outlet_Sales"].mean()

    reorder_point = avg_sales * 0.5

    col1, col2 = st.columns(2)

    col1.metric(

        "Average Weekly Demand",
        f"{avg_sales:.0f}"

    )

    col2.metric(

        "Reorder Point",
        f"{reorder_point:.0f}"

    )

    if current_stock < reorder_point:

        st.error("Reorder Needed")

        recommended = avg_sales * 2

        st.success(

            f"Recommended Order Quantity: {recommended:.0f}"

        )

    else:

        st.success("Inventory Level Healthy")


# ---------------- PRICE OPTIMIZER ----------------

elif page == "Price Optimizer":

    st.title("AI Price Optimization")

    prices = np.linspace(50,300,100)

    test = pd.DataFrame({

        "Item_MRP":prices,
        "Item_Weight":[10]*100,
        "Item_Visibility":[0.05]*100,
        "Profit":[12]*100

    })

    demand = model.predict(test)

    revenue = prices * demand

    best_price = prices[np.argmax(revenue)]

    st.metric(

        "Optimal Price",
        f"₹{best_price:.0f}"

    )

    fig = px.line(

        x=prices,
        y=revenue,
        labels={"x":"Price","y":"Revenue"},
        title="Revenue Optimization Curve"

    )

    st.plotly_chart(fig, width="stretch")


# ---------------- MODEL ANALYTICS ----------------

elif page == "Model Analytics":

    st.title("Model Performance")

    col1, col2 = st.columns(2)

    col1.metric("R² Score", f"{r2:.2f}")
    col2.metric("MAE", f"{mae:.2f}")

    st.markdown("---")

    st.subheader("Feature Importance")

    importance = pd.DataFrame({

        "Feature":features,
        "Importance":model.feature_importances_

    }).sort_values(

        "Importance",
        ascending=False

    )

    fig = px.bar(

        importance,
        x="Importance",
        y="Feature",
        orientation="h"

    )

    st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    st.subheader("Explainable AI (SHAP)")

    sample = df[features].sample(100)

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(sample)

    shap.summary_plot(

        shap_values,
        sample,
        show=False

    )

    st.pyplot()