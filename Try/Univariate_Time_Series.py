import numpy as np
import pandas as pd
from datetime import datetime
from pandas_datareader import data as web
import yfinance as yf
import statsmodels.tsa.api as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from scipy.stats import probplot, moment
from numpy.linalg import LinAlgError
from arch import arch_model
from arch.univariate import ConstantMean, GARCH, Normal
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

def plot_correlogram(x, lags=None, title=None):
    lags = min(10, int(len(x)/5)) if lags is None else lags
    with sns.axes_style('whitegrid'):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
        x.plot(ax=axes[0][0], title='Residuals')
        x.rolling(21).mean().plot(ax=axes[0][0], c='k', lw=1)
        q_p = np.max(q_stat(acf(x, nlags=lags), len(x))[1])
        stats = f'Q-Stat: {np.max(q_p):>8.2f}\nADF: {adfuller(x)[1]:>11.2f}'
        axes[0][0].text(x=.02, y=.85, s=stats, transform=axes[0][0].transAxes)
        probplot(x, plot=axes[0][1])
        mean, var, skew, kurtosis = moment(x, moment=[1, 2, 3, 4])
        s = f'Mean: {mean:>12.2f}\nSD: {np.sqrt(var):>16.2f}\nSkew: {skew:12.2f}\nKurtosis:{kurtosis:9.2f}'
        axes[0][1].text(x=.02, y=.75, s=s, transform=axes[0][1].transAxes)
        plot_acf(x=x, lags=lags, zero=False, ax=axes[1][0])
        plot_pacf(x, lags=lags, zero=False, ax=axes[1][1])
        axes[1][0].set_xlabel('Lag')
        axes[1][1].set_xlabel('Lag')
        fig.suptitle(title, fontsize=14)
        sns.despine()
        fig.tight_layout()
        fig.subplots_adjust(top=.9)
        plt.savefig(title)

def make_components(time_data_key = "IPGMFN",start='1990',end = '2017-12-31'):
    data = web.DataReader(time_data_key , data_source='fred', start=start, end = end).squeeze().dropna()
    components = tsa.seasonal_decompose(data, model='additive')
    ts = (data.to_frame('Original')
          .assign(Trend=components.trend)
          .assign(Seasonality=components.seasonal)
          .assign(Residual=components.resid))
    with sns.axes_style('white'):
        ts.plot(subplots=True, figsize=(14, 8), title=['Original Series', 'Trend Component', 'Seasonal Component','Residuals'], legend=False)
        plt.suptitle('Seasonal Decomposition', fontsize=14)
        sns.despine()
        plt.tight_layout()
        plt.subplots_adjust(top=.91);
        plt.savefig('Components of a Continuos Time Series with proper Frequency')

def draw_log_diff(stock,data,data_log,data_log_diff):
    with sns.axes_style('dark'):
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 8),squeeze=False)

        data.plot(ax=axes[0][0],
                    title=stock+'  Composite Index')
        axes[0][0].text(x=.03,
                        y=.85,
                        s=f'ADF: {tsa.adfuller(data.dropna())[1]:.4f}',
                        transform=axes[0][0].transAxes)
        axes[0][0].set_ylabel('Index')

        data_log.plot(ax=axes[1][0],
                        sharex=axes[0][0])
        axes[1][0].text(x=.03, y=.85,
                        s=f'ADFl: {tsa.adfuller(data_log.dropna())[1]:.4f}',
                        transform=axes[1][0].transAxes)
        axes[1][0].set_ylabel('Log')

        data_log_diff.plot(ax=axes[2][0],
                             sharex=axes[0][0])
        axes[2][0].text(x=.03, y=.85,
                        s=f'ADF: {tsa.adfuller(data_log_diff.dropna())[1]:.4f}',
                        transform=axes[2][0].transAxes)
        axes[2][0].set_ylabel('Log, Diff')

        sns.despine()
        fig.tight_layout()
        fig.align_ylabels(axes)
        plt.savefig(stock+' Index, Log, Log_Diff')

def build_ARIMA(stock,data_log_diff,window = 120):
    results = {}
    y_true = data_log_diff.iloc[window:]

    for p in range(3,6):
        for d in range(0,1):
            for q in range(3,6):
                print('#########################################################################################################################################################################################################################################################')
                print(p, q, d)
                print('#########################################################################################################################################################################################################################################################')

                aic, bic, y_pred = [], [], []
                convergence_error = 0
                stationarity_error = 0

                for T in range(window, len(data_log_diff)):
                    train_set = data_log_diff.iloc[T-window:T]
                    try: model = tsa.ARIMA(endog=train_set, order=(p,d, q)).fit()
                    except LinAlgError: convergence_error += 1
                    except ValueError: stationarity_error += 1

                    forecast, _, _ = model.forecast(steps=1)
                    y_pred.append(forecast[0])
                    aic.append(model.aic)
                    bic.append(model.bic)

                result = (pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).replace(np.inf, np.nan).dropna())
                rmse = np.sqrt(mean_squared_error(y_true=result.y_true, y_pred=result.y_pred))
                results[(p, d, q)] = [rmse, np.mean(aic), np.mean(bic), convergence_error, stationarity_error]

    arima_results = pd.DataFrame(results).T
    arima_results.columns = ['RMSE', 'AIC', 'BIC', 'convergence', 'stationarity']
    arima_results.index.names = ['p','d', 'q']
    print(arima_results)
    with pd.HDFStore(stock+'arima.h5') as store:
        store.put(stock+'arima', arima_results)
    return arima_results

def build_GARCH(stock,data_log_diff,window = 5 * 252):
    ## clipping/trimming extreme volatiliy returns
    data = data_log_diff.clip(lower=data_log_diff.quantile(.05), upper=data_log_diff.quantile(.95))
    T = len(data)
    results = {}
    for p in range(1, 5):
        for q in range(1, 5):
            print(f'{p} | {q}')
            result = []
            for s, t in enumerate(range(window, T-1)):
                train_set = data.iloc[s: t]
                test_set = data.iloc[t+1]  # 1-step ahead forecast
                model = arch_model(y=train_set,mean='Constant', lags=0, vol='Garch', p=p, o=0, q=q, power=2.0, dist='Normal').fit(disp='off')
                forecast = model.forecast(horizon=1)
                mu = forecast.mean.iloc[-1, 0]
                var = forecast.variance.iloc[-1, 0]
                result.append([(test_set-mu)**2, var])
            df = pd.DataFrame(result, columns=['y_true', 'y_pred'])
            results[(p, q)] = np.sqrt(mean_squared_error(df.y_true, df.y_pred))
            s = pd.Series(results)
            s.index.names = ['p', 'q']
            s = s.unstack().sort_index(ascending=False)
            sns.heatmap(s, cmap='Blues', annot=True, fmt='.4f')
            plt.title('Out-of-Sample RMSE');
            plt.savefig(stock+'_GARCH_Selection_matrix_using_RMSE_1_step_predic.png')
    return

def main():
    end = '2015-01-01'
    start = '2007-01-01'
    interval = '1d'
    symbols = ["^NSEI"]
    EDA = True
    Build_Arima = True
    garch = False
    Build_Garch = False

    for stock in symbols:
        tic = yf.Ticker(stock)
        data = tic.history(start=start,end=end, interval=interval).dropna()
        close = data['Close']
        close_log = np.log(close)
        close_log_diff = close_log.diff().dropna()

        if EDA:
            draw_log_diff(stock,close,close_log,close_log_diff)
            plot_correlogram(pd.Series(np.random.normal(size=10000)),lags=100,title= 'White Noise')
            plot_correlogram(close_log_diff, lags=100, title=stock + ' Daily returns (Log_diff)')
            plot_correlogram(close_log_diff.sub(close_log_diff.mean()).pow(2), lags=100, title=stock+' Daily Volatility')

        if Build_Arima:
            arima_results = build_ARIMA(stock,close_log_diff,window = 4*252)
        else:
            arima_results = pandas.read_hdf(stock+'arima.h5')

        # chose configuration with lowest RMSE and BIC also use the more parsimonious configuration:
        best_p, best_d, best_q = arima_results.rank().loc[:, ['RMSE', 'BIC']].mean(1).idxmin()
        best_arima_model = tsa.ARIMA(endog=close_log_diff, order=(best_p, best_d, best_q)).fit()
        print(best_arima_model.summary())
        plot_correlogram(best_arima_model.resid, title = stock +str(best_p)+','+str(best_d)+','+str(best_q)+ ' ARIMA model residuals (should be like White Noise)')
        plot_correlogram(best_arima_model.resid.sub(best_arima_model.resid.mean()).pow(2), title = stock + str(best_p)+','+str(best_d)+','+str(best_q)+' ARIMA model residual Square (Checking Volatility)')

        if garch:
            if Build_Garch:
                build_GARCH(stock,close_log_diff,window = 5 * 252)

            best_p, best_q = 2 , 2 # Take From Image of GARCH_Selection_matrix_using_RMSE_1_step_predic
            garch_data = close_log_diff.clip(lower=close_log_diff.quantile(.05), upper=close_log_diff.quantile(.95))
            best_garch_model = arch_model(y=garch_data,mean='Constant', lags=0, vol='Garch', p=best_p, o=0, q=best_q, power=2.0, dist='Normal').fit(disp='off')
            print(best_garch_model.summary())

            plt.clf()
            fig = best_model.plot(annualize='D')
            fig.set_size_inches(12, 8)
            fig.tight_layout();
            plt.savefig('GARCH Standardized Residuals and Anualized Conditional Volatilities')
            plot_correlogram(best_model.resid.dropna(), lags=250, title=stock + 'GARCH Residuals')
main()
