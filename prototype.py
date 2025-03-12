#Author Andrew Jeske
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import time
import random
import requests
import os
from retrivingStock import retrieveStock
from main import load_model
from main_training import train_and_evaluate_model
from Features import calculate_features
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from plots import plot_feature_importance, plot_predictions

# Initialize session state variables
if "is_training" not in st.session_state:
    st.session_state.is_training = False
    
if "training_data" not in st.session_state:
    st.session_state.training_data = None
    
if "selected_stock_for_training" not in st.session_state:
    st.session_state.selected_stock_for_training = None

def stop_training():
    with open("stop_training_flag.txt", "w") as f:
        f.write("STOP")
    st.warning("Stopping training... This may take a moment.")
    st.session_state.is_training = False

ALPHA_VANTAGE_API_KEY = "YOUR_API_KEY"
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('📈 Stock Prediction Model')

@st.cache_data
def load_tickers():
    df_tickers = pd.read_csv('stockTickers.csv')
    return df_tickers

@st.cache_data
def fetch_stock_description_yahoo(ticker):
    """Fetches stock description from Yahoo Finance with caching and error handling."""
    try:
        time.sleep(random.uniform(1, 3))
        stock = yf.Ticker(ticker)
        info = stock.info

        if info and "longBusinessSummary" in info:
            return info["longBusinessSummary"]
    except Exception:
        pass

    return None

@st.cache_data
def fetch_stock_description_alpha_vantage(ticker):
    """Fetch stock description from Alpha Vantage as a fallback."""
    try:
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        data = response.json()

        return data.get("Description", "Stock description is currently unavailable.")
    except Exception:
        return "Stock description is currently unavailable."

@st.cache_data
def fetch_stock_description(ticker):
    """Try Yahoo Finance first, then fallback to Alpha Vantage or CSV."""
    description = fetch_stock_description_yahoo(ticker)
    if description:
        return description

    # Fallback: Load local CSV with descriptions
    try:
        df = pd.read_csv('stock_descriptions.csv')
        description = df[df['Ticker'] == ticker]['Description'].values
        return description[0] if len(description) > 0 else fetch_stock_description_alpha_vantage(ticker)
    except Exception:
        return fetch_stock_description_alpha_vantage(ticker)

stock_df = load_tickers()
stock_options = stock_df['Ticker'].tolist()
selected_stock = st.selectbox("Select dataset for prediction", stock_options)

selected_stock_name = stock_df[stock_df['Ticker'] == selected_stock]['Name'].values[0]

stock_description = fetch_stock_description(selected_stock)

st.subheader(f"About {selected_stock_name}")
st.write(stock_description)


mode = st.radio("Choose mode", ["Train Model", "Load & Predict"])

status_container = st.container()
with status_container:
    st.write(f"Training status: {'Active' if st.session_state.is_training else 'Inactive'}")

if st.button("🔍 Execute", key="execute_button"): #ABSOLUTELY DO NOT TOUCH
    df = retrieveStock(selected_stock, end_date="2024-01-01")
    if df is not None and not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df_with_features = calculate_features(df)
        df_with_features[['Close', 'Date']] = df[['Close', 'Date']]
        
        if mode == "Train Model": #SAVING DATA FOR STREAMLIT
            st.session_state.training_data = df_with_features
            st.session_state.selected_stock_for_training = selected_stock
            st.session_state.is_training = True
            st.rerun()
            
        if mode == "Load & Predict":
            st.subheader("Loading Trained Model...")
            model = load_model(selected_stock)
            if model:
                df_with_features['Actual_Price_Change'] = (df_with_features['Close'].diff() > 0).astype(int)
                df_with_features.dropna(subset=['Actual_Price_Change'], inplace=True)
                feature_columns = [col for col in df_with_features.columns if col not in ['Date', 'Close', 'Actual_Price_Change']]
                X_test = df_with_features[feature_columns]
                
                predictions = model.predict(X_test)
                results_df = pd.DataFrame({
                    'Date': df_with_features['Date'],
                    'Actual_Close': df_with_features['Close'],
                    'Actual_Price_Change': df_with_features['Actual_Price_Change'],
                    'Predicted_Price_Change': predictions
                })
                
                st.subheader("Prediction Results")
                st.write(results_df[['Date', 'Actual_Close', 'Actual_Price_Change', 'Predicted_Price_Change']].tail())
                
                st.subheader("Classification Report")
                report = classification_report(results_df['Actual_Price_Change'], results_df['Predicted_Price_Change'], output_dict=True)
                st.json(report)
                
                st.subheader("Confusion Matrix")
                matrix = confusion_matrix(results_df['Actual_Price_Change'], results_df['Predicted_Price_Change'])
                st.write(matrix)
                plot_predictions(results_df)
            else:
                st.error("No trained model found.")

if st.session_state.is_training:
    if st.button("Stop Training", key="stop_training_btn"):
        stop_training()

if st.session_state.is_training and st.session_state.training_data is not None:
    st.subheader(f"Training Model for {st.session_state.selected_stock_for_training}...")
    
    progress_placeholder = st.empty()
    results_placeholder = st.empty()
    
    try:
        results_df, best_model, feature_columns = train_and_evaluate_model(
            st.session_state.training_data, 
            st.session_state.selected_stock_for_training
        )
        if os.path.exists("stop_training_flag.txt"):
            os.remove("stop_training_flag.txt")
            st.warning("Training was stopped by user.")
        else:
            st.success(f"Training completed for {st.session_state.selected_stock_for_training}")
            
        if results_df is not None and not results_df.empty:
            st.write("Training Results:")
            st.write(results_df.tail())
            
            if best_model:
                st.success(f"Best model saved with score: {results_df['Score'].max() if 'Score' in results_df.columns else 'N/A'}")
                if hasattr(best_model, 'feature_importances_'):
                    st.subheader("Feature Importance")
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    importance = pd.DataFrame({
                        'Feature': feature_columns,
                        'Importance': best_model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    sns.barplot(x='Importance', y='Feature', data=importance, ax=ax)
                    st.pyplot(fig)
    except Exception as e:
        st.error(f"Error during training: {str(e)}")
    finally:
        st.session_state.is_training = False
        if os.path.exists("stop_training_flag.txt"):
            os.remove("stop_training_flag.txt")