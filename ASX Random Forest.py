"""
Random Forest Stock Prediction

This script trains a Random Forest model to predict the next day's
stock movement (up/down) using historical stock data.

Author: Sue Wang
Date: 2026-01-06
Data: S&P ASX 200 Most Recent Data
"""

#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score


#constants
DATA = 'S&P_ASX 200 Historical Data.csv' #name of any predownloaded historical data csv
TRAIN_RATIO = 2/3
YF_TICKER = "^AXJO" #yahoo finance ticker of any stock
YF_START = "2000-01-01"
H = 20          # forward horizon in trading days
THRESH = 0.01   # "investable" if there is at least +1% return over H days


#functions
#use a predownloaded csv instead, allo
def load_data(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst = True)

    Amounts = ['Price', 'Open', 'High', 'Low']
    for x in Amounts:
        df[x] = df[x].astype(str).str.replace(',', '', regex=False)
        df[x] = pd.to_numeric(df[x], errors='coerce') #make unparseable values NaN

    df['Vol.'] = df['Vol.'].apply(parse_volume)
    df = df.dropna(subset=["Date", "Price", "Open", "High", "Low"]) #drop rows with missing or unparseable data
    df = df.sort_values('Date').reset_index(drop=True)
    return df

#downloads data from Yahoo Finance
def fetch_data(ticker=YF_TICKER, start=YF_START):
    df = yf.Ticker(ticker).history(start=start, auto_adjust=False)

    if df is None or df.empty:
        raise RuntimeError("No data found")

    #flatten columns if they are multiIndex 
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    else:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df = df.reset_index()

    # read all dates
    if "Date" not in df.columns:
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        elif "index" in df.columns:
            df = df.rename(columns={"index": "Date"})
        else:
            raise RuntimeError(f"Couldn't find a Date column. Columns: {df.columns.tolist()}")

    df = df.rename(columns={"Close": "Price", "Volume": "Vol."})

    df = df[["Date", "Price", "Open", "High", "Low", "Vol."]].copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for col in ["Price", "Open", "High", "Low", "Vol."]:
        s = df[col]
        if isinstance(s, pd.DataFrame):   # if duplicate columns created a DataFrame
            s = s.iloc[:, 0]
        df[col] = pd.to_numeric(s, errors="coerce")

    df = df.dropna(subset=["Date", "Price", "Open", "High", "Low"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


#clean data
def parse_volume(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x.endswith('B'):
        return float(x[:-1]) * 1_000_000_000
    elif x.endswith('M'):
        return float(x[:-1]) * 1_000_000
    elif x.endswith('K'):
        return float(x[:-1]) * 1_000
    else:
        return float(x)

#build predictive features
def create_features(df):
    df["Return_1"] = df["Price"].pct_change()
    df["Return_5"] = df["Price"].pct_change(5)
    df["MA_5"] = df["Price"].rolling(5).mean()
    df["Price_vs_MA5"] = df["Price"] / df["MA_5"] - 1
    df["Volatility_5"] = df["Return_1"].rolling(5).std()
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    # forward return over H trading days P(t+H)/P(t) - 1
    df["FwdRet_H"] = df["Price"].shift(-H) / df["Price"] - 1

    # classification target, label 1 if forward return is greater than 1%
    df["Target"] = (df["FwdRet_H"] >= THRESH).astype(int)

    df = df.dropna().reset_index(drop=True)
    df = pd.get_dummies(df, columns=["DayOfWeek"], drop_first=True)
    return df

#retrieves most recent data
def latest_features(df_raw, X_cols):
    df_raw = df_raw.sort_values('Date').reset_index(drop=True)

    if len(df_raw) < 6:
        raise RuntimeError("Not enough rows to compute Return_5 (need at least 6).")

    last_price = df_raw['Price'].iloc[-1]

    return_1 = df_raw["Price"].iloc[-1] / df_raw["Price"].iloc[-2] - 1
    return_5 = df_raw["Price"].iloc[-1] / df_raw["Price"].iloc[-6] - 1

    ma_5_today = df_raw['Price'].iloc[-5:].mean()
    price_vs_ma5 = df_raw["Price"].iloc[-1] / ma_5_today - 1

    returns_1_series = df_raw["Price"].pct_change()
    volatility_5 = returns_1_series.iloc[-5:].std()

    day_of_week = df_raw["Date"].iloc[-1].dayofweek

    X_latest = pd.DataFrame({
        "Return_1": [return_1],
        "Return_5": [return_5],
        "MA_5": [ma_5_today],
        "Price_vs_MA5": [price_vs_ma5],
        "Volatility_5": [volatility_5],
        "DayOfWeek_1": [1 if day_of_week == 1 else 0],
        "DayOfWeek_2": [1 if day_of_week == 2 else 0],
        "DayOfWeek_3": [1 if day_of_week == 3 else 0],
        "DayOfWeek_4": [1 if day_of_week == 4 else 0],
        "DayOfWeek_5": [1 if day_of_week == 5 else 0],
        "DayOfWeek_6": [1 if day_of_week == 6 else 0],
    })

    X_latest = X_latest.reindex(columns=X_cols, fill_value=0)

    latest_date = df_raw['Date'].iloc[-1]
    return X_latest, latest_date, last_price


# main script
if __name__ == "__main__":
    df = fetch_data()
    df.to_csv(DATA, index=False)

    df_train = create_features(df.copy())

    X_cols = [
        "Return_1", "Return_5", "MA_5", "Price_vs_MA5", "Volatility_5",
        "DayOfWeek_1", "DayOfWeek_2", "DayOfWeek_3", "DayOfWeek_4", "DayOfWeek_5", "DayOfWeek_6"
    ]

    X = df_train.reindex(columns=X_cols, fill_value=0)
    y = df_train["Target"]
    realized = df_train["FwdRet_H"]  

    # time-ordered train/test split
    train_size = int(len(df_train) * TRAIN_RATIO)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    realized_test = realized[train_size:]

    rf = RandomForestClassifier(
        n_estimators=1000,
        max_features="sqrt",
        max_depth=8,
        min_samples_leaf=50,
        random_state=42
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    # classification matrics
    print(f"Accuracy (H={H}, Threshold={THRESH:.2%}): {accuracy_score(y_test, y_pred):.3f}")
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=3))

    # ROC AUC requires both classes present in y_test
    if len(np.unique(y_test)) == 2:
        auc = roc_auc_score(y_test, y_prob)
        print(f"ROC AUC: {auc:.3f}")
    else:
        print("ROC AUC: N/A (only one class present in test set)")

    # --- Investing-style evaluation on test set ---
    # "Trade" when model predicts Target=1, and use the realized H-day forward return
    take_trade = (y_pred == 1)
    coverage = take_trade.mean()

    if take_trade.sum() > 0:
        avg_realized = realized_test.values[take_trade].mean()
        med_realized = np.median(realized_test.values[take_trade])
    else:
        avg_realized = np.nan
        med_realized = np.nan

    baseline_avg = realized_test.values.mean()

    print(f"\nTrade coverage (predicted buy signals): {coverage:.3f}")
    print(f"Avg realized fwd return when trading: {avg_realized:.3%}")
    print(f"Median realized fwd return when trading: {med_realized:.3%}")
    print(f"Baseline avg fwd return (always invested): {baseline_avg:.3%}")

    # --- Predict the next H days (signal + probability) ---
    X_latest, latest_date, last_price = latest_features(df, X_cols)

    pred_signal = rf.predict(X_latest)[0]
    pred_prob = rf.predict_proba(X_latest)[0][1]

    action = "BUY" if pred_signal == 1 else "NO TRADE"
    print(f"\nLatest date in data: {latest_date.date()}  (Close={last_price:.2f})")
    print(f"Model signal for next {H} trading days (>= {THRESH:.2%}): {action}")
    print(f"Probability of meeting threshold: {pred_prob:.3f}")

    # plots (unchanged)
    features = ['Open', 'High', 'Low', 'Price', 'Vol.']

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()

    for i, col in enumerate(features):
        sns.lineplot(x=df["Date"], y=df[col], ax=axes[i])
        axes[i].set_title(col)

    fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.show()
