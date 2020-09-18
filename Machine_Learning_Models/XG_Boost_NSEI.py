import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf

import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

def build_xg_boost(stock, X, y, trian_fraction = 0.8):
    train_length = int(len(X)*trian_fraction)
    X_train = X[:train_length]
    X_test = X[train_length:]
    y_train = y[:train_length]
    y_test = y[train_length:]

    model = XGBClassifier(max_depth=100, n_estimators=150)
    kfold = KFold(n_splits=5, random_state=7)
    results = cross_val_score(model, X_train, y_train, cv=kfold)
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

    model.fit(X_train, y_train)
    xgboost.plot_importance(model, max_num_features=7)
    plt.savefig('importace.png')

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    array = confusion_matrix(y_test, y_pred)
    heatmap = pd.DataFrame(array, index=['Short', 'Long'], columns=['Short', 'Long'])
    plt.figure(figsize=(5, 4))
    sns.heatmap(heatmap, annot=True, cmap='Greens', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('Actual vs Predicted.png')
    return model

def main():
    end = '2014-12-31'
    start = '2011-01-01'
    interval = '1d'
    symbols = ["^NSEI"]
    EDA = False
    Build_XgBoost = True
    trian_fraction = 0.8

    for stock in symbols:
        tic = yf.Ticker(stock)
        data = tic.history(start=start,end=end, interval=interval).dropna()
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']

        close_log = np.log(close)
        close_log_diff = close_log.diff().dropna()
        high_log = np.log(high)
        high_log_diff = high_log.diff().dropna()
        low_log = np.log(low)
        low_log_diff = low_log.diff().dropna()
        volume_log = np.log(volume)
        volume_log_diff = volume_log.diff().dropna()

        use_data =  pd.concat([close_log_diff,high_log_diff,low_log_diff,volume_log_diff],axis=1)
        use_data.columns = ['close','high','low','volume']
        predictor_list = ['close','high','low','volume']

        for r in range(5, 21, 5):

            use_data['close_rolling_mean_'+str(r)] = use_data.close.rolling(r).sum()
            # use_data['close_rolling_std_'+str(r)] = use_data.close.rolling(r).std()
            predictor_list.append('close_rolling_mean_'+str(r))
            # predictor_list.append('close_rolling_std_'+str(r))

            use_data['high_rolling_mean_'+str(r)] = use_data.high.rolling(r).sum()
            # use_data['high_rolling_std_'+str(r)] = use_data.high.rolling(r).std()
            predictor_list.append('high_rolling_mean_'+str(r))
            # predictor_list.append('high_rolling_std_'+str(r))

            use_data['low_rolling_mean_'+str(r)] = use_data.low.rolling(r).sum()
            # use_data['low_rolling_std_'+str(r)] = use_data.low.rolling(r).std()
            predictor_list.append('low_rolling_mean_'+str(r))
            # predictor_list.append('low_rolling_std_'+str(r))

            use_data['volume_rolling_mean_'+str(r)] = use_data.volume.rolling(r).sum()
            # use_data['volume_rolling_std_'+str(r)] = use_data.volume.rolling(r).std()
            predictor_list.append('volume_rolling_mean_'+str(r))
            # predictor_list.append('volume_rolling_std_'+str(r))

        use_data['close_next_day'] = use_data.close.shift(-1)
        use_data['actual_signal'] = np.where(use_data.close_next_day > 0, 1, -1)
        use_data = use_data.dropna()

        X = use_data[predictor_list]
        y = use_data['actual_signal']

        if Build_XgBoost:
            xg_boost_results = build_xg_boost(stock, X, y, trian_fraction = trian_fraction)
main()
