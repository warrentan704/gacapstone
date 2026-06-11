import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load files
# -----------------------------
churn_model = joblib.load("churn_model.pkl")
churn_features = joblib.load("churn_features.pkl")

df_churn_risk = pd.read_csv("customer_churn_risk.csv")
df_churn_drivers = pd.read_csv("churn_risk_drivers.csv")


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

st.title('Customer Churn Prediction App')

st.info (
    "This app estimates whether a customer is likely to become inactive in the next 30 days "
    "based on recent shopping behavior."
)


# -----------------------------
# Helper function
# -----------------------------
def assign_risk_level(probability):
    if probability >= 0.60:
        return "High Risk"
    elif probability >= 0.30:
        return "Medium Risk"
    else:
        return "Low Risk"


def explain_risk_level(risk_level):
    if risk_level == "High Risk":
        return "This customer is likely to become inactive soon and should be prioritized for retention."
    elif risk_level == "Medium Risk":
        return "This customer shows some signs of inactivity and may benefit from targeted engagement."
    else:
        return "This customer currently shows relatively low churn risk."


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Choose an option",
    [
        "Customer Lookup",
        "Manual Prediction",
        "Churn Risk Drivers"
    ]
)


# -----------------------------
# Page 1: Customer Lookup
# -----------------------------
if page == "Customer Lookup":

    st.header("Customer Lookup")

    st.write(
        "Search for an existing household to view its predicted churn risk."
    )

    household_list = sorted(df_churn_risk["household_key"].unique())

    selected_household = st.selectbox(
        "Select household key",
        household_list
    )

    customer = df_churn_risk[
        df_churn_risk["household_key"] == selected_household
    ].iloc[0]

    churn_probability = customer["churn_probability"]
    risk_level = customer["risk_level"]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Household Key",
        int(selected_household)
    )

    col2.metric(
        "Churn Probability",
        f"{churn_probability:.1%}"
    )

    col3.metric(
        "Risk Level",
        risk_level
    )

    st.info(explain_risk_level(risk_level))

    st.subheader("Customer Behavior Summary")

    display_cols = [
        "recency",
        "tenure_days",
        "purchase_frequency",
        "discount_rate",
        "median_basket_sales",
        "median_basket_quantity",
        "median_unique_products"
    ]

    customer_summary = customer[display_cols].to_frame("Value")

    customer_summary.index = [
        "Days since last purchase",
        "Customer tenure in days",
        "Purchase frequency",
        "Discount reliance",
        "Typical basket spend",
        "Typical basket quantity",
        "Typical product variety"
    ]

    st.dataframe(customer_summary, use_container_width=True)


# -----------------------------
# Page 2: Manual Prediction
# -----------------------------
elif page == "Manual Prediction":

    st.header("Manual Churn Prediction")

    st.write(
        "Enter a customer's recent shopping behavior to estimate their churn risk."
    )

    col1, col2 = st.columns(2)

    with col1:
        recency = st.number_input(
            "Days since last purchase",
            min_value=0,
            max_value=365,
            value=30,
            help="How many days ago the customer last shopped. Higher values usually mean higher churn risk."
        )

        tenure_days = st.number_input(
            "Customer tenure in days",
            min_value=1,
            max_value=800,
            value=365,
            help="Number of days between the customer's first and most recent purchase."
        )

        purchase_frequency = st.number_input(
            "Purchase frequency",
            min_value=0.0,
            max_value=5.0,
            value=0.10,
            step=0.01,
            help="Average number of shopping trips per day during the customer's observed history."
        )

        discount_rate = st.slider(
            "Discount reliance",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.01,
            help="Share of sales value represented by discounts. Higher values suggest stronger discount sensitivity."
        )

    with col2:
        median_basket_sales = st.number_input(
            "Typical basket spend",
            min_value=0.0,
            max_value=500.0,
            value=25.0,
            step=1.0,
            help="The customer's typical basket value."
        )

        median_basket_quantity = st.number_input(
            "Typical basket quantity",
            min_value=0.0,
            max_value=200.0,
            value=10.0,
            step=1.0,
            help="The typical number of items bought per shopping trip."
        )

        median_unique_products = st.number_input(
            "Typical product variety",
            min_value=0.0,
            max_value=200.0,
            value=8.0,
            step=1.0,
            help="The typical number of different products bought per shopping trip."
        )

    input_data = pd.DataFrame([{
        "recency": recency,
        "tenure_days": tenure_days,
        "purchase_frequency": purchase_frequency,
        "discount_rate": discount_rate,
        "median_basket_sales": median_basket_sales,
        "median_basket_quantity": median_basket_quantity,
        "median_unique_products": median_unique_products
    }])

    input_data = input_data[churn_features]

    if st.button("Predict Churn Risk"):

        churn_probability = churn_model.predict_proba(input_data)[:, 1][0]
        risk_level = assign_risk_level(churn_probability)

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        col1.metric(
            "Churn Probability",
            f"{churn_probability:.1%}"
        )

        col2.metric(
            "Risk Level",
            risk_level
        )

        st.info(explain_risk_level(risk_level))


# -----------------------------
# Page 3: Churn Risk Drivers
# -----------------------------
elif page == "Churn Risk Drivers":

    st.header("Churn Risk Drivers")

    st.write(
        "These are the behavioral factors that the model uses to estimate churn risk."
    )

    drivers = df_churn_drivers.copy()

    drivers["driver"] = drivers["feature"].replace({
        "recency": "Days since last purchase",
        "tenure_days": "Customer tenure",
        "purchase_frequency": "Purchase frequency",
        "discount_rate": "Discount reliance",
        "median_basket_sales": "Typical basket spend",
        "median_basket_quantity": "Typical basket quantity",
        "median_unique_products": "Typical product variety"
    })

    st.dataframe(
        drivers[
            [
                "driver",
                "coefficient",
                "effect_on_churn",
                "abs_coefficient"
            ]
        ],
        use_container_width=True
    )

    st.write(
        "Positive coefficients increase churn risk. Negative coefficients reduce churn risk."
    )
