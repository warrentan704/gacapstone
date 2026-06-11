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

st.image("tesco_logo.jpg", width=180)

st.title("Customer Churn Prediction")
st.info(
    "This app estimates whether a TESCO customer is likely to become inactive in the next 30 days "
    "based on recent shopping behavior."
)


# -----------------------------
# Helper functions
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


def friendly_customer_summary(customer):
    summary = pd.Series({
        "Days since last purchase": customer["recency"],
        "Customer tenure in days": customer["tenure_days"],
        "Typical shopping trips per month": customer["purchase_frequency"] * 30,
        "Typical discount reliance (%)": customer["discount_rate"] * 100,
        "Typical basket spend": customer["median_basket_sales"],
        "Typical basket quantity": customer["median_basket_quantity"],
        "Typical product variety": customer["median_unique_products"]
    })

    return summary.to_frame("Value")


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

    customer_summary = friendly_customer_summary(customer)

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

        shopping_trips_per_month = st.number_input(
            "Typical shopping trips per month",
            min_value=0,
            max_value=30,
            value=3,
            step=1,
            help="How many times the customer usually shops in a month."
        )

        discount_reliance_pct = st.slider(
            "Typical discount reliance (%)",
            min_value=0,
            max_value=100,
            value=10,
            step=1,
            help="Roughly what percentage of the customer's shopping value comes from discounts."
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

    # Convert intuitive management inputs into model inputs
    purchase_frequency = shopping_trips_per_month / 30
    discount_rate = discount_reliance_pct / 100

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
        "purchase_frequency": "Shopping frequency",
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
