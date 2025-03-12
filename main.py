
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import  f1_score, classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from Features import calculate_features
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit
from retrivingStock import retrieveStock
from Features import calculate_features
from sklearn.model_selection import TimeSeriesSplit,GridSearchCV
import matplotlib.pyplot as plt
from bokeh.plotting import figure
from bokeh.embed import json_item
import json, os, io
from sklearn.pipeline import Pipeline
from plots import plot_feature_importance, plot_predictions
from sklearn.ensemble import RandomForestClassifier
from data_cleaning import clean_data
import time
import joblib
"""Author: Kyle Nguyen"""
"""Stocks tickers that work: GOOG"""
def load_model(stock_symbol):
    model_file = f"saved_models/{stock_symbol}.pkl"
    return joblib.load(model_file)


def train_and_evaluate_model(df):
    start_time = time.time()
    df['Price_Change'] = (df['Close'].diff() > 0).astype(int)
    df.dropna(inplace=True)
    feature_columns = [col for col in df.columns if col not in ['Date', 'Close', 'Price_Change']]
    X = df[feature_columns]
    y = df['Price_Change']
    dates = df['Date']

    #Pipeline
    pipeline = Pipeline([
        ('model', RandomForestClassifier(random_state=42))
    ])

    #hyperparameters grid
    params = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [10, 5, None],
        'model__min_samples_split': [5, 30],
        'model__min_samples_leaf': [2, 20],
        'model__max_features': ['sqrt'],
        'model__class_weight': ['balanced']
    }
    tscv = TimeSeriesSplit(n_splits=8)

    #grid search
    grid_search = GridSearchCV(
        pipeline, params, cv=tscv, scoring='f1', n_jobs=-1, verbose=1
    )
    grid_search.fit(X, y)

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.4f}")

    #Cross-validation
    results = []
    best_model = grid_search.best_estimator_

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        best_model.fit(X_train, y_train)
        predictions = best_model.predict(X_test)
        fold_accuracy = f1_score(y_test, predictions)
        print(f"Fold {fold + 1} f1: {fold_accuracy:.4f}")

        results.append({
            'Fold': fold + 1,
            'Actual': y_test.values,
            'Predicted': predictions,
            'Dates': dates.iloc[test_index].values
        })

    #final eval
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print("\nClassification Report on Test Set:")
    print(classification_report(y_test, y_pred))

    combined_results = pd.DataFrame({
        'Date': np.concatenate([result['Dates'] for result in results]),
        'Actual_Price_Change': np.concatenate([result['Actual'] for result in results]),
        'Predicted_Price_Change': np.concatenate([result['Predicted'] for result in results])
    })

    print(f"Total training time: {time.time() - start_time:.2f} seconds")
    return combined_results, best_model, feature_columns


def main():
    name = input("Enter Stock Symbol or Q to end Program: ")
    if name.upper() == 'Q':
        print("Exiting program.")
        return
    model = load_model(name)
    
    if model == None:
        print("No trained model found")
        return
    
    try:
        df = retrieveStock(name, end_date="2024-01-01") 
        if df is None or df.empty:
            print(f"Error: No data available for {name}.")
            return
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df_with_features = calculate_features(df)
        
        common_indices = df_with_features.index
        df_aligned = df.iloc[common_indices].copy()
        
        df_with_features['Close'] = df_aligned['Close']
        df_with_features['Date'] = df_aligned['Date']
        
        df_with_features['Actual_Price_Change'] = (df_aligned['Close'].diff() > 0).astype(int)
        df_with_features.dropna(subset=['Actual_Price_Change'], inplace=True)
        feature_columns = [col for col in df_with_features.columns if col not in ['Date', 'Close', 'Actual_Price_Change']]
        X_test = df_with_features[feature_columns]

        #make prediction with the trained model
        predictions = model.predict(X_test)
        
        #results df
        results_df = pd.DataFrame({
            'Date': df_with_features['Date'],
            'Actual_Close': df_with_features['Close'],
            'Actual_Price_Change': df_with_features['Actual_Price_Change'],
            'Predicted_Price_Change': predictions
        })

        print(results_df.tail())
        
        print("\nClassification Report:")
        print(classification_report(results_df['Actual_Price_Change'], results_df['Predicted_Price_Change']))
        
        #confusion matrix
        print("\nConfusion Matrix:")
        matrix = confusion_matrix(results_df['Actual_Price_Change'], results_df['Predicted_Price_Change'])
        print(matrix)
        print("\nConfusion Matrix Interpretation:")
        print(f"True Negative: {matrix[0][0]} - Correctly predicted price going down")
        print(f"False Positive: {matrix[0][1]} - Predicted price going up when it actually went down")
        print(f"False Negative: {matrix[1][0]} - Predicted price going down when it actually went up")
        print(f"True Positive: {matrix[1][1]} - Correctly predicted price going up")
                
        plot_predictions(results_df)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
