# ================================================================
# Project: FRAUDSHIELD - Credit Card Fraud Detection
# Author: Mehul Khandelwal
# Streamlit App for Real-Time Fraud Detection
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from io import BytesIO
import base64

# ================= Load Model & Scaler =================
model_data = joblib.load("fraudshield_best_model.pkl")
model = model_data["model"]
threshold = model_data["threshold"]
feature_columns = model_data["feature_names"]  # 30 features: Time, V1-V28, Amount

scaler = joblib.load("scaler.pkl")

# Load training medians for PCA features (V1-V28)
pca_medians = joblib.load("pca_medians.pkl")

# ================= Streamlit Page Config =================
st.set_page_config(
    page_title="FraudShield | Secure Credit Card Fraud Detection 🛡️",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= Header =================
st.markdown("""
    <div style="text-align: center;">
        <h1 style="font-size: clamp(36px, 5vw, 55px);">FraudShield: Credit Card Fraud Detection 💳</h1>
        <p style="font-size: 22px;">
            Detect fraudulent transactions using a <b>Machine Learning model</b> trained on real-world credit card data.
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# ================= Model Insights =================
st.markdown(
    """
    <div style="font-size:20px; line-height:1.7; margin-top:25px; margin-bottom:25px; padding:15px; 
                background-color:#f9f9fb; border:1px solid #e0e0e0; border-radius:10px;">
        <p><b>🧠 Model Insights</b></p>
        <p>1. Powered by a <b>Random Forest Classifier</b>, trained with <b>SMOTEENN</b> to handle class imbalance.<br>
        2. Optimized to detect rare but critical fraudulent transactions with improved precision.</p>
        <p style="margin-top:18px; color:#d62828; font-size:18px;">
        ⚠️ <b>Disclaimer:</b> This app is for demonstration and educational purposes only. 
        It is <strong>not</strong> intended for direct financial or production deployment.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ================= Sidebar =================
st.markdown("""
    <style>
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: #f0f4ff;  /* light blue background */
    }

    /* Sidebar title and text */
    [data-testid="stSidebar"] h2 {
        color: #1a237e;  /* deep navy */
        font-weight: 700;
    }

    [data-testid="stSidebar"] label {
        font-size: 16px;
        font-weight: 500;
        color: #212121;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.header("⚙️ Input Options")
mode = st.sidebar.radio("Select Input Mode:", ["Single Transaction Examples", "Batch CSV Upload"])

# ================= Risk Level Function =================
def risk_level(prob):
    if prob < 0.2:
        return "**Low Risk 🟢**"
    elif prob < 0.5:
        return "**Medium Risk 🟡**"
    else:
        return "**High Risk 🔴**"

# ================= Example Transactions =================
example_df = pd.read_csv("single_transaction_examples.csv")

# ================= Single Transaction Example Mode =================
if mode == "Single Transaction Examples":
    st.sidebar.subheader("📌 Select Example Transaction")

    def label_mapper(val):
        return "Not Fraud" if val == 0 else "Fraud"

    selected_idx = st.sidebar.selectbox(
        "Choose Transaction",
        example_df.index,
        format_func=lambda x: f"Transaction {x+1} (Label: {label_mapper(example_df.loc[x,'Label'])})"
    )

    user_df = example_df.loc[[selected_idx], feature_columns]
    user_df[["Time", "Amount"]] = scaler.transform(user_df[["Time", "Amount"]])

    # ================= Button Styling =================
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            font-weight: 700;
            font-size: 18px;
            color: white;
            border-radius: 10px;
            padding: 0.6em 1.2em;
            border: none;
            background: linear-gradient(90deg, #d62828 0%, #f77f00 100%);
            transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.3s ease;
        }
        div.stButton > button:first-child:hover {
            transform: translateY(-2px);
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.25);
            background: linear-gradient(90deg, #f77f00 0%, #d62828 100%);
        }
        </style>
    """, unsafe_allow_html=True)

    if st.button("🚀 Predict Fraud Risk"):
        prob_fraud = model.predict_proba(user_df)[:,1][0]
        prediction = int(prob_fraud > threshold)

        if prediction == 1:
            st.error(f"🚨 This transaction is likely **FRAUDULENT**!\n\nFraud Probability: {prob_fraud*100:.2f}%")
        else:
            st.success(f"✅ This transaction appears **LEGITIMATE**.\n\nFraud Probability: {prob_fraud*100:.2f}%")

        level = risk_level(prob_fraud)
        st.markdown(f"### 📊 Risk Level: **{level}**")

        # 🎯 Riskometer Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob_fraud * 100,
            title={'text': "Fraud Probability (%)", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#d62828"},  # Fraud red (bar pointer)
                'steps': [
                    {'range': [0, 20], 'color': "#4caf50"},   # green
                    {'range': [20, 40], 'color': "#cddc39"},  # lime
                    {'range': [40, 60], 'color': "#ffeb3b"},  # yellow
                    {'range': [60, 80], 'color': "#ff9800"},  # orange
                    {'range': [80, 100], 'color': "#f44336"}  # red
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)



# ================= Batch CSV Upload Mode =================
elif mode == "Batch CSV Upload":
    st.sidebar.subheader("📂 Upload CSV File for Multiple Transactions")
    uploaded_file = st.file_uploader("Choose CSV File", type=["csv"])

    st.info(
        """
        ⚠️ **Important Instructions for CSV Upload**  
        - The file **must include** these columns:  
          `Time, V1, V2, ..., V28, Amount`  
        - `V1–V28` are PCA-transformed features from the dataset.  
        - If only **Time** and **Amount** are provided, missing `V1–V28` values  
          will be **filled with training medians** (⚠️ this may reduce accuracy).  
        - ✅ **Best Practice:** Always provide the complete set of features for reliable predictions.
        """
    )

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)

            # Fill missing PCA features
            for i in range(1, 29):
                col = f"V{i}"
                if col not in df_input.columns:
                    df_input[col] = pca_medians[i-1]

            for col in ["Time", "Amount"]:
                if col not in df_input.columns:
                    df_input[col] = 0.0

            df_input[["Time", "Amount"]] = scaler.transform(df_input[["Time", "Amount"]])
            df_input = df_input[feature_columns]

            # Predictions
            y_prob = model.predict_proba(df_input)[:, 1]
            y_pred = (y_prob > threshold).astype(int)
            df_input["Fraud_Probability"] = y_prob
            df_input["Prediction"] = y_pred
            df_input["Prediction_Label"] = df_input["Prediction"].apply(lambda x: "Fraud" if x==1 else "Legit")

            st.success(f"✅ Predictions completed for {len(df_input)} transactions")

            # Fraud highlighting
            def highlight_fraud(row):
                return ['background-color: #ff9999' if row["Prediction_Label"]=="Fraud" else '' for _ in row]

            st.markdown('<p class="section-header">🔍 Prediction Results</p>', unsafe_allow_html=True)
            st.dataframe(df_input.head(200).style.apply(highlight_fraud, axis=1))
            st.info("📌 Showing first 200 rows with fraud highlighting. Download full results below.")

            # Save styled Excel
            def to_excel_styled(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    df.to_excel(writer, index=False, sheet_name="Predictions")
                    workbook = writer.book
                    worksheet = writer.sheets["Predictions"]
                    red_format = workbook.add_format({"bg_color": "#FF9999"})
                    fraud_rows = df.index[df["Prediction_Label"]=="Fraud"].tolist()
                    for row in fraud_rows:
                        worksheet.set_row(row+1, None, red_format)
                return output.getvalue()

            excel_data = to_excel_styled(df_input)

            st.download_button(
                label="📥 Download Predictions (Excel with Fraud Highlighting)",
                data=excel_data,
                file_name="fraud_predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            # Fraud distribution
            st.markdown('<p class="section-header">📊 Fraud Probability Distribution</p>', unsafe_allow_html=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=df_input["Fraud_Probability"]*100, nbinsx=50, marker_color='indianred'))
            fig2.update_layout(
                xaxis_title="Fraud Probability (%)",
                yaxis_title="Number of Transactions",
                bargap=0.2
            )
            st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error("❌ The uploaded file could not be processed. Please check that it has the correct format and required columns.")
            st.exception(e)  
            
# ================= Footer =================
def add_footer():
    logo_path = "logo.png"  # your logo file
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .footer {{
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background: linear-gradient(90deg, #e3f2fd, #f8f9fa); /* soft blue to gray */
            color: #222;
            padding: 12px 25px;
            font-size: 18px;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-top: 1px solid #ddd;
            z-index: 1000;
        }}
        .footer-text {{
            margin: 0 auto;
            text-align: center;
            flex: 1;
        }}
        .footer-logo img {{
            height: 45px;
            background: rgba(255,255,255,0.7);
            border-radius: 8px;
            padding: 2px;
        }}

        [data-testid="stSidebar"][aria-expanded="true"] ~ .main .footer {{
            margin-left: 300px; /* default sidebar width */
            width: calc(100% - 300px);
        }}
        [data-testid="stSidebar"][aria-expanded="false"] ~ .main .footer {{
            margin-left: 0;
            width: 100%;
        }}

        </style>

        <div class="footer">
            <div class="footer-text">✨ Built with care by <b>Mehul Khandelwal</b></div>
            <div class="footer-logo">
                <img src="data:image/png;base64,{logo_base64}" alt="Logo">
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Call footer at end
add_footer()
