import pandas as pd
import yfinance as yf
import math as m
import numpy as np
import pickle
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from arch import arch_model
import matplotlib.pyplot as plt

def tsplot(y, lags=None, title='remove', figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))

        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
        plt.savefig(title+".png")
    return

def EDA(diff_data):
    white_noise = np.random.normal(size=1000)
    tsplot(white_noise, lags=30,title='White Noise') # plot of discrete white noise
    tsplot(diff_data, lags=30,title='diff_data') # plot of our Time Series

def get_best_arima(diff_data):
    best_aic = np.inf
    best_order = None; best_mdl = None
    pq_rng = range(5); d_rng = range(2)

    for i in pq_rng:
        for d in d_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = smt.ARIMA(diff_data, order=(i,d,j)).fit(method='mle', trend='nc')
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic; best_order = (i, d, j); best_mdl = tmp_mdl
                except:
                    continue
    print('\n\n\n aic: ' + str(best_aic) +'\n order:' + str(best_order))
    tsplot(best_mdl.resid, lags=30, title = 'Best ARIMA')
    return best_mdl,best_order,best_aic


def forcast(best_mdl,forcast_start_date,n_steps=21,alpha=0.01):
    # Create a 21 day forecast of stock returns
    f, err95, ci95 = best_mdl.forecast(steps=n_steps)
    _, err_alpha, ci_alpha = best_mdl.forecast(steps=n_steps, alpha=alpha)
    idx = pd.date_range(forcast_start_date, periods=n_steps, freq='D')
    fc_95 = pd.DataFrame(np.column_stack([f, ci95]), index=idx, columns=['forecast', 'lower_ci_95', 'upper_ci_95'])
    fc_alpha = pd.DataFrame(np.column_stack([ci_alpha]), index=idx, columns=['lower_ci_alpha', 'upper_ci_alpha'])
    fc_all = fc_95.combine_first(fc_alpha)
    return fc_all

def main():
    np.random.seed(1)
    end = '2015-01-01'
    start = '2007-01-01'
    symbols = ["^NSEI"]
    n_steps = 21

    for stock in symbols:
        tic = yf.Ticker(stock)
        data = tic.history(start=start,end=end)
        diff_data = np.diff(np.log(data['Close']))
        print(' For Difference of log of time series\n Mean  = ' + str(diff_data.mean()) + ' \n Variance = ' + str(diff_data.var()) +' \n Standard Deviation = ' + str(diff_data.std()))
        EDA(diff_data)
        diff_data = pd.DataFrame(diff_data,index=data.index[1:])
        try:
            pickle_in = open(str(stock) + "best.pickle","rb")
            [best_mdl,best_order,best_aic] = pickle.load(pickle_in)
        except:
            best_mdl,best_order,best_aic=get_best_arima(diff_data)
            pickle_out = open(str(stock) + "best.pickle","wb")
            pickle.dump([best_mdl,best_order,best_aic], pickle_out)
            pickle_out.close()

        predictions = forcast(best_mdl,data.index[-1],n_steps,alpha=0.01)
        in_sample_prediction = best_mdl.predict(diff_data[-100:].index[0],diff_data[-100:].index[-1])

        plt.clf()
        plt.style.use('bmh')
        plt.plot(diff_data[-100:],label='Áctual Difference') # True Values from train
        plt.plot(in_sample_prediction, label = 'In Sample Predictions') # predicted Values from train data
        plt.plot(predictions['forecast'],label = 'Forecast') # Predicted values for future
        plt.fill_between(predictions.index, predictions.lower_ci_95, predictions.upper_ci_95, color='gray', alpha=0.7)
        plt.fill_between(predictions.index, predictions.lower_ci_alpha, predictions.upper_ci_alpha, color='gray', alpha=0.2)
        plt.title(str(n_steps)+' Day ' +str(stock)+ ' Return Forecast using ARIMA')
        plt.legend()
        plt.savefig(str(stock)+' Forcast for next '+str(n_steps)+ 'days' )

        # # Plot 21 day forecast for stock returns
        # fig = plt.figure(figsize=(10,8))
        # ax = plt.gca()
        #
        # ts = diff_data.iloc[-500:].copy()
        # ts.plot(ax=ax, label=str(stock) + ' Returns')
        #
        # # in sample prediction
        # pred = best_mdl.predict(ts.index[0], ts.index[-1])
        # pred.plot(ax=ax, style='r-', label='In-sample prediction')
        #
        # # styles = ['b-', '0.2', '0.75', '0.2', '0.75']
        # # fc_all.plot(ax=ax, style=styles)
        # # plt.xlim(0, 500+n_steps)
        # # plt.fill_between(fc_all.index, fc_all.lower_ci_95, fc_all.upper_ci_95, color='gray', alpha=0.7)
        # # plt.fill_between(fc_all.index, fc_all.lower_ci_alpha, fc_all.upper_ci_alpha, color='gray', alpha=0.2)
        # # plt.title(str(n_steps)+' Day ' +str(stock)+ ' Return Forecast using ARIMA')
        # # plt.legend(loc='best', fontsize=10)
        # #
main()
