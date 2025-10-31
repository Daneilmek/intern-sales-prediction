# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# Set page configuration
st.set_page_config(
    page_title="Sales Prediction App",
    page_icon="📊",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    model = joblib.load('sales_prediction_model.pkl')
    return model

model = load_model()

# App title and description
st.title("📊 Advertising Sales Prediction")
st.markdown("""
This app predicts **Sales** based on advertising budgets using a Linear Regression model.
Adjust the sliders to see how different advertising strategies affect sales!
""")

# Sidebar for user inputs
st.sidebar.header("🎯 Advertising Budget Input")

# Input sliders
tv_budget = st.sidebar.slider(
    "TV Advertising Budget ($ thousands)", 
    min_value=0.0, 
    max_value=300.0, 
    value=150.0, 
    step=1.0
)

radio_budget = st.sidebar.slider(
    "Radio Advertising Budget ($ thousands)", 
    min_value=0.0, 
    max_value=50.0, 
    value=25.0, 
    step=1.0
)

newspaper_budget = st.sidebar.slider(
    "Newspaper Advertising Budget ($ thousands)", 
    min_value=0.0, 
    max_value=100.0, 
    value=25.0, 
    step=1.0
)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("💰 Prediction Results")
    
    # Make prediction
    input_data = np.array([[tv_budget, radio_budget, newspaper_budget]])
    prediction = model.predict(input_data)[0]
    
    # Display prediction
    st.metric(
        label="Predicted Sales",
        value=f"${prediction:.2f} thousands",
        delta=f"${prediction-14:.2f} vs average"  # Assuming average sales around 14
    )
    
    # Budget allocation pie chart
    st.subheader("📊 Budget Allocation")
    budgets = [tv_budget, radio_budget, newspaper_budget]
    labels = ['TV', 'Radio', 'Newspaper']
    
    fig, ax = plt.subplots()
    ax.pie(budgets, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures pie is circular
    st.pyplot(fig)

with col2:
    st.subheader("📈 Impact Analysis")
    
    # Feature importance based on model coefficients
    coefficients = model.coef_
    features = ['TV', 'Radio', 'Newspaper']
    
    importance_df = pd.DataFrame({
        'Channel': features,
        'Impact per $1k': coefficients
    }).sort_values('Impact per $1k', ascending=False)
    
    st.dataframe(importance_df, use_container_width=True)
    
    # Bar chart of impacts
    fig, ax = plt.subplots()
    bars = ax.barh(importance_df['Channel'], importance_df['Impact per $1k'])
    ax.set_xlabel('Sales Impact per $1,000 spent')
    ax.bar_label(bars, fmt='%.3f')
    st.pyplot(fig)

# Additional features
st.markdown("---")
st.subheader("🔍 Model Insights")

col3, col4 = st.columns(2)

with col3:
    st.write("**Optimal Budget Strategy:**")
    st.write("""
    - TV advertising has the highest impact
    - Radio provides good secondary impact  
    - Newspaper has the lowest ROI
    - Consider focusing on TV + Radio combination
    """)

with col4:
    st.write("**Business Recommendations:**")
    st.write(f"""
    - Current budget: ${tv_budget + radio_budget + newspaper_budget:.0f}k
    - Expected return: ${prediction:.2f}k sales
    - ROI: {(prediction/(tv_budget + radio_budget + newspaper_budget)*100):.1f}%
    """)

# Batch prediction feature
st.markdown("---")
st.subheader("📋 Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV file for multiple predictions", 
    type=['csv'],
    help="File should have columns: TV, Radio, Newspaper"
)

if uploaded_file is not None:
    try:
        # Read uploaded file
        batch_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(batch_data.head())
        
        # Check if required columns exist
        required_cols = ['TV', 'Radio', 'Newspaper']
        if all(col in batch_data.columns for col in required_cols):
            # Make predictions
            predictions = model.predict(batch_data[required_cols])
            batch_data['Predicted_Sales'] = predictions
            
            st.write("Predictions Results:")
            st.dataframe(batch_data)
            
            # Download results
            csv = batch_data.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="sales_predictions.csv",
                mime="text/csv"
            )
        else:
            st.error(f"File must contain columns: {required_cols}")
            
    except Exception as e:
        st.error(f"Error processing file: {e}")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit & Scikit-learn*")