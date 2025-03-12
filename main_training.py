"""Author: Kyle Nguyen"""
"""_summary_
This main.py is for model training purposes, it will be used to train the model with the given data and the results will be logged into a results folder, 
the best model will also be saved as a .pkl
"""
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from retrivingStock import retrieveStock
from Features import calculate_features
from sklearn.model_selection import TimeSeriesSplit,GridSearchCV, ParameterSampler,RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from debug import time_counter
import time, sys, threading, json
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import random,joblib,os
from pynput import keyboard
from scipy.stats import randint
isRunning = True

def exit_program():
    global isRunning
    while isRunning:
        userinput = input("Press q then enter to exit.".strip().lower())
        if userinput == 'q':
            print("\nProgram Exiting...")
            isRunning = False
            break


def stopwatch():
    start_time = time.time()
    try:
        while isRunning:
            elapsed_time = time.time() - start_time
            sys.stdout.write(f"\rTraining Time: {elapsed_time:.2f} seconds")
            sys.stdout.flush()
            time.sleep(0.1)
            if os.path.exists("stop_training_flag.txt"):#check for stop flag
                break
    except:
        pass
    finally:
        elapsed_time = time.time() - start_time
        sys.stdout.write(f"\rTraining Time: {elapsed_time:.2f} seconds (Completed)")
        sys.stdout.flush()


def randomGrid(): #updated grid for RandomizedSearchCV
    return {
        'n_estimators': randint(50, 501),
        'max_depth': [None, 10, 15, 20, 30,40, 50, 70, 100, 150, 200],
        'min_samples_split': randint(2, 51),
        'min_samples_leaf': randint(1, 31),
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', None]
    }


def train_and_evaluate_model(df, stock_symbol):
    global isRunning
    isRunning = True
    
    os.makedirs("ResultsCSV", exist_ok=True)
    logFile = f"ResultsCSV/{stock_symbol}_results.csv"
    
    #results df to return for Streamlit
    results_df = pd.DataFrame(columns=['Iteration', 'Score', 'Parameters'])
    print(f"Start training for {stock_symbol}... Click 'Stop Training' to stop\n")
    df['Price_Change'] = (df['Close'].diff() > 0).astype(int)
    df.dropna(inplace=True)
    feature_col = [col for col in df.columns if col not in ['Date', 'Close', 'Price_Change']]
    X, y = df[feature_col], df['Price_Change']
    tscv = TimeSeriesSplit(n_splits=8)
    best_model = None
    best_score = 0
    best_parameters = None
    stopwatch_thread = threading.Thread(target=stopwatch, daemon=True)
    stopwatch_thread.start()
    iteration = 0
    try:
        while isRunning:
            iteration += 1
            if os.path.exists("stop_training_flag.txt"):
                print("\nStop signal received. Ending training...")
                try:#resetting stop flag
                    os.remove("stop_training_flag.txt")
                except:
                    pass
                isRunning = False
                break

            params = randomGrid()
            try:
                GridSearch = RandomizedSearchCV(
                    RandomForestClassifier(random_state=42),
                    params,
                    n_iter=3,
                    cv=tscv,
                    scoring='f1',
                    n_jobs=-1,
                    random_state=int(time.time()),
                    verbose=0
                )
                
                GridSearch.fit(X, y)
                
                if GridSearch.best_score_ > best_score:
                    best_model = GridSearch.best_estimator_
                    best_parameters = GridSearch.best_params_
                    best_score = GridSearch.best_score_
                    print(f"\nNew Best score for {stock_symbol} is {best_score}")
                    print(f"\nNew Best Parameters: {best_parameters}")
                
                #REMINDER: For deprecated append method
                new_row = pd.DataFrame({
                    'Iteration': [iteration],
                    'Score': [GridSearch.best_score_],
                    'Parameters': [str(GridSearch.best_params_)]
                })
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                
            except KeyboardInterrupt:
                print("Training interrupted")
                isRunning = False
                break
                
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        isRunning = False
        if best_model is not None:
            os.makedirs("saved_models", exist_ok=True)
            model_filename = f"saved_models/{stock_symbol}.pkl"                    
            joblib.dump(best_model, model_filename)
            print(f"\nBest model saved as {model_filename}.")
    
    print("Training Stopped")
    return results_df, best_model, feature_col


def main(): #For Training
    global stock_symbol
    stock_symbol = input("Enter Stock Symbol: ").upper()

    try:
        print(f"\nTraining started for {stock_symbol}...\n")
        
        df = retrieveStock(stock_symbol, end_date="2024-01-01")
        if df is None or df.empty:
            print(f"Error: No data retrieved for {stock_symbol}. Check if the stock symbol is correct or if data is available.")
            return
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df_with_features = calculate_features(df)
        df_with_features[['Close', 'Date']] = df[['Close', 'Date']]
        exit_thread = threading.Thread(target=exit_program, daemon=True)
        exit_thread.start()
        #use return values from train_and_evaluate_model
        _, best_model, _ = train_and_evaluate_model(df_with_features, stock_symbol)
    except Exception as e:
        print(f"Error with {stock_symbol}: {e}")