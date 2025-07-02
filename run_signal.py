# 📓 Sensex Enriched Gamma + Macro Strategy Real-Time Trade Signal (Single Trade Mode)

# 📌 SECTION 1: Install Required Packages
!pip install yfinance scikit-learn xgboost lightgbm imbalanced-learn ta requests --quiet

# 📌 SECTION 2: Import Libraries
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from datetime import datetime, timedelta

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# 📌 SECTION 3: Load Historical Data

def load_data(ticker, period="5y", interval="1d"):
    data = yf.download(ticker, period=period, interval=interval)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    return data

sensex_df = load_data("^BSESN")
us_df = load_data("^GSPC")
crude_df = load_data("BZ=F")
vix_df = load_data("^VIX")
dxy_df = load_data("DX-Y.NYB")
bond_df = load_data("^TNX")
copper_df = load_data("HG=F")
shanghai_df = load_data("000001.SS")
btc_df = load_data("BTC-USD")

# 📌 SECTION 4: Technical + Macro Feature Engineering

def enrich_indicators(df):
    df["Returns"] = df["Close"].pct_change()
    df["ATR"] = df["High"].rolling(14).max() - df["Low"].rolling(14).min()
    df["ATR_mean"] = df["ATR"].rolling(20).mean()
    df["Volatility_Filter"] = df["ATR"] > df["ATR_mean"]
    df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()
    df["MACD"] = MACD(close=df["Close"]).macd_diff()
    df["ADX"] = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"]).adx()
    return df

sensex_df = enrich_indicators(sensex_df)
us_df["Returns"] = us_df["Close"].pct_change()
crude_df["Returns"] = crude_df["Close"].pct_change()
vix_df["Change"] = vix_df["Close"].pct_change()
dxy_df["Change"] = dxy_df["Close"].pct_change()
bond_df["Change"] = bond_df["Close"].pct_change()
copper_df["Change"] = copper_df["Close"].pct_change()
shanghai_df["Change"] = shanghai_df["Close"].pct_change()
btc_df["Change"] = btc_df["Close"].pct_change()

# Merge + Lag

df = sensex_df[["Close", "Returns", "Volatility_Filter", "RSI", "MACD", "ADX"]].copy()
df = df.join(us_df[["Returns"]].rename(columns={"Returns": "US_Returns"}))
df = df.join(crude_df[["Returns"]].rename(columns={"Returns": "Crude_Returns"}))
df = df.join(vix_df[["Change"]].rename(columns={"Change": "VIX_Change"}))
df = df.join(dxy_df[["Change"]].rename(columns={"Change": "DXY_Change"}))
df = df.join(bond_df[["Change"]].rename(columns={"Change": "BOND_Change"}))
df = df.join(copper_df[["Change"]].rename(columns={"Change": "Copper_Change"}))
df = df.join(shanghai_df[["Change"]].rename(columns={"Change": "Shanghai_Change"}))
df = df.join(btc_df[["Change"]].rename(columns={"Change": "BTC_Change"}))

df["Rolling_3d"] = df["Returns"].rolling(3).mean()
df["Rolling_5d"] = df["Returns"].rolling(5).mean()
df["Lag_1"] = df["Returns"].shift(1)
df["Lag_2"] = df["Returns"].shift(2)

# 📌 SECTION 5: Target + Modeling Prep

future_return = df["Returns"].shift(-1)
df["Target"] = np.where(future_return > 0.003, 1, 0)

features = df.drop(columns=["Target", "Close", "Returns"]).dropna()
target = df["Target"].loc[features.index]

selector = SelectKBest(score_func=f_classif, k=20)
X_selected = selector.fit_transform(features, target)
selected_columns = features.columns[selector.get_support()]
X = features[selected_columns].values
y = target.values

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(X_res, y_res):
    X_train, X_test = X_res[train_idx], X_res[test_idx]
    y_train, y_test = y_res[train_idx], y_res[test_idx]

xgb = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.8, use_label_encoder=False, eval_metric='logloss')
lgb = LGBMClassifier(n_estimators=150, max_depth=4, learning_rate=0.05)
logreg = LogisticRegression()

ensemble = VotingClassifier(
    estimators=[("xgb", xgb), ("lgb", lgb), ("logreg", logreg)],
    voting="soft",
    weights=[3, 2, 1]
)
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
probs = ensemble.predict_proba(X_test)[:, 1]

print("\n🧠 Ensemble Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("ROC-AUC:", round(roc_auc_score(y_test, probs), 4))

# 📌 SECTION 6: Single High-Confidence Trade Suggestion

latest = features[selected_columns].iloc[-1:]
prob = ensemble.predict_proba(latest)[0][1]
label = ensemble.predict(latest)[0]
latest_spot = sensex_df["Close"].iloc[-1]
base_strike = round(latest_spot / 100) * 100
expiry = (pd.to_datetime(df.index[-1]) + pd.DateOffset(days=(7 - pd.to_datetime(df.index[-1]).weekday()))).strftime("%d %b")

option_type = "CE" if label == 1 else "PE"
confidence = "High" if prob > 0.8 else ("Moderate" if prob > 0.6 else "Low")

if prob > 0.6:
    print("\n📈 Live Trade Suggestion:")
    print(f"  → {expiry} {int(base_strike)} {option_type}")
    print("  → Entry Time: NOW | Exit: Monitor for 15:15 or SL")
    print(f"  → Confidence: {confidence} | Estimated Success: {round(prob*100,2)}%")
else:
    print("\n⚠️ No Trade Suggested — Confidence too Low:")
    print(f"  → Predicted Probability: {round(prob*100,2)}%")
    print("  → Waiting for better signal in next cycle.")
