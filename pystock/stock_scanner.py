# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 17:15:24 2020

@author: I Kit Cheng

"""

from pandas_datareader import data as pdr
import yahoo_fin.stock_info as si
import yahoo_fin.news as news
from pandas import ExcelWriter
import yfinance as yf
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta, date
import re
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import mplfinance as mpf
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split


class Stock():
    def __init__(self, ticker, download_data=False, ):
        self.ticker = ticker
        self.technical = None
        self.df = None # technical data for last 30 days (training)
        self.y = None # buy sell labels for last 30 days (training)
        self.quote = None
        self.fundamental = None
        self.financials = None
        self.stats = None
        self.trailingPE = None
        self.EPS = None
        self.marketCap = None
        self.TotalDebtEquity = None
        self.shortRatio = None
        self.institutionPct = None
        self.earnings = None
        self.earnings_up_count = None # number of times earnings went up in last 4 quarters
        self.earnings_up_latest = None # 1 if latest earnings was higher than previous
        self.errorCount = 0

        if download_data:
            self.download_data()
     
    def download_data(self):
        """ Get initial data for stock"""
        self.get_technical_data()
        self.get_fundamental_data()
        
        
    def get_technical_data(self, start_date=datetime.now()-timedelta(days=365),
                           end_date=None, interval='1d'):
        """ 
        Downloads historical stock price data into a pandas data frame.  Interval
        must be "1d", "1wk", "1mo", or "1m" for daily, weekly, monthly, or minute data.
        Intraday minute data is limited to 7 days.
        """
        self.technical = si.get_data(self.ticker, start_date=start_date,
                                     end_date=end_date, interval=interval)
        return self.technical
        
    def get_buy_sell_label(self):
        """ Get buy sell label based on whether tomorrow 
        is green or red (win_label). 
        Idea: use these labels to train a classifier to predict probability 
        of buying or selling today (before market closes). """
        assert self.df is not None, "No stock data."       
        buy_sell_label = ['buy' if self.df.iloc[i+1].win_label==1  else 'sell' for i in range(len(self.df)-1)]
        buy_sell_label.append('None')
        self.df['label'] = buy_sell_label
                
    def get_fundamental_data(self):
        """ all fundamental data """
        try: self.fundamental = si.get_quote_data(self.ticker)
        except: pass
        try: self.quote = si.get_quote_table(self.ticker)
        except: pass
        try: self.stats = si.get_stats(self.ticker)
        except: pass
        try: self.earnings = si.get_earnings(self.ticker)
        except: pass
        try: self.financials = si.get_financials(self.ticker)
        except: pass
        
    
        # Individual fundamental factors

        try: self.earnings_up_count = sum((self.earnings['quarterly_revenue_earnings'].earnings.diff()>0).iloc[1:])
        except: self.errorCount += 1
    
        try: self.earnings_up_latest = int((self.earnings['quarterly_revenue_earnings'].earnings.diff()>0).iloc[-1])
        except: self.errorCount += 1
    
        try: self.trailingPE = self.quote['PE Ratio (TTM)']
        except: self.errorCount += 1
        
        try: self.EPS = self.quote['EPS (TTM)']
        except: self.errorCount += 1
        
        try: self.marketCap = self.fundamental['marketCap']
        except: self.errorCount += 1 
        
        try: self.TotalDebtEquity = self.stats[
            self.stats.Attribute == 'Total Debt/Equity (mrq)'].iloc[0,1]
        except: self.errorCount += 1
        
        try: self.shortRatio = float(self.stats[
            self.stats.Attribute.str.contains('Short Ratio')].iloc[0,1])
        except: self.errorCount += 1
        
        try: self.institutionPct = float(self.stats[
            self.stats.Attribute.str.contains('Held by Institutions')].iloc[0,1][:-1])
        except: self.errorCount += 1
    
    
    def get_news_data(self):
        self.news = news.get_yf_rss(self.ticker)

        

    def get_first_day_vol_explosion(self, threshold=2):
        """ Get first day of high volume (>2 std from mean in prev 30 days). 
        Returns number of days since first high volume day, and the date.
        """
        assert self.df is not None, "No stock data."        

        for i in range(len(self.df)):
            if self.df.vol_zscore.iloc[-(i+1)]>threshold:
                pass
            else:
                if i==0:
                    print('Not high volume stock.')
                return i, self.df.index[-i]
            
            
    def plot_vol_zscore_close_high(self):
        """ Plot volume zscore and close, high price """
        assert self.technical is not None, "No stock data."        
        fig, ax = plt.subplots(3,1)
        ax[0].plot(self.df.vol_zscore, label='vol') 
        ax[0].legend()
        ax[0].set_ylabel('Z-score')
        ax[1].plot(self.df.close, label='close')
        ax[1].plot(self.df.high, label='high')
        ax[1].set_ylabel('Price')
        ax[1].legend()
        
        ax[2].plot(self.df.label)
        ax[2].set_ylabel('buy or sell')
        return ax
 
    
    def plot_chart(self):
        """ Plot stock chart with volume, 50 and 200 moving averages"""
        assert self.technical is not None, "No stock data."
        mpf.plot(self.technical, type='candle', mav=(50,200), volume=True)
    
    def get_train_data(self):
        """ Build multivariate timeseries dataset for training ML model.
        Use last 30 days data. """
        
        self.df = self.get_technical_data(start_date=datetime.now()-timedelta(days=30),
                           end_date=date.today(), interval='1d')
        
        self.df['vol_mean'] = self.df.volume.iloc[:-1].mean()
        
        self.df['vol_std'] = self.df.volume.iloc[:-1].std()
        
        self.df['vol_zscore'] = (self.df.volume-self.df.vol_mean)/self.df.vol_std
        
        self.df['win_label']=(self.df['close'].diff()>0).astype(int)
        
        self.get_buy_sell_label()
        
        self.df['marketCap'] = self.marketCap
        
        self.df['institutionPct'] = self.institutionPct
        
        if self.shortRatio is None:
            self.df['shortRatio'] = 0
        else:
            self.df['shortRatio'] = self.shortRatio
        
        if self.earnings_up_count is None:
            self.df['earnings_up_count'] = 0
        else:
            self.df['earnings_up_count'] = self.earnings_up_count
         
        if self.earnings_up_count is None:
            self.df['earnings_up_latest'] = 0
        else:
            self.df['earnings_up_latest'] = self.earnings_up_latest
        
    def preprocess_train_data(self, seq_len=7, stride=1):
        """ Preprocess train data. """
        
        # log prices so prices on same scale while preserving the hierarchy
        self.X = np.concatenate((np.log(self.df[['open','high','low','close']].values),
                                 self.df[['vol_zscore']]),
                                axis=1)
        
        self.X, self.y = generate_timeseries_dataset(self.X,
                                             self.df['label'].values,
                                             seq_len,
                                             len(self.df),
                                             sampling_rate=1,
                                             stride=stride)[0]
        
        # Stretch each window of 7 days into a single row
        self.X = self.X.reshape(self.X.shape[0], -1)
        
        # Add fundamental data to end of each row
        # self.X = np.concatenate((self.X, np.zeros((len(self.X),5))),
        #                         axis=1)
        
        # self.X[:, -5:] = [np.log(self.marketCap), self.institutionPct/100, self.shortRatio,
        #                   self.earnings_up_count, self.earnings_up_latest]
        
        # fill any None or nans with 0.
        self.X[np.isnan(self.X)] = 0
        
        return self.X, self.y
        
        
def generate_timeseries_dataset(data, targets, length, batch_size, sampling_rate=1, stride=1, start_index=0, center=False):
    """ 
    Utility class for generating batches of temporal data, with option to 
    choose output to be the centre of the window; center = True.
    TimeseriesGenerator generates data window from t to t+dt and target at t+dt+1 
    
    For center = True,
    Expect total number of windows = len(data) - (window_length - 1)
    Example: len(data) = 20, window_length = 10, n_windows = 20 - (10-1) = 11
    """
    if center==True:
        pad_len = int(length/2)
        # shift labels down by half window length such that the first target is 
        # the label at the center of the first window
        targets = np.pad(targets, (pad_len,0), 'constant', constant_values=0)[:-pad_len+1].reshape(-1,1) 
        # pad 1 row of zeros at the bottom of data matrix to match length of targets: (top,bottom),(left,right)
        data = np.pad(data, ((0,1),(0,0)), 'constant', constant_values=0)
    else:
        # shift labels down by 1 such that the first target is 
        # the label at the end of the first window
        targets = np.pad(targets, (1,0), 'constant', constant_values=0).reshape(-1,1) #[:-1]
        # pad 1 row of zeros at the bottom of data matrix to match length of targets: (top,bottom),(left,right)
        data = np.pad(data, ((0,1),(0,0)), 'constant', constant_values=0)
        
    data_gen = TimeseriesGenerator(data, targets,
                                  length=length, sampling_rate=sampling_rate,
                                  stride=stride,
                                  start_index=start_index,
                                  batch_size=batch_size)
    return data_gen

        
class StockScanner(Stock):
    def __init__(self):
        #self.tickers_dow = si.tickers_dow()
        self.tickers_ftse100 = si.tickers_ftse100() 
        self.tickers_ftse250 = si.tickers_ftse250()
        self.tickers_nasdaq = si.tickers_nasdaq()
        self.tickers_sp500 = si.tickers_sp500()
        self.tickers_other = si.tickers_other()
        self.tickers_all = self.get_all_tickers()
        self.clean_LSE_tickers()
            
        self.tickers_high_vol = []
        self.stock_picks = {}
        
        # Train data for all my_stocks
        self.X = None
        self.y = None
        self.label = None
    
    def clean_LSE_tickers(self):
        """
        Clean the ftse100 and ftse250 stock tickers by adding .L at the end. 
        This allows getting data for these tickers from yahoo finance. 
        """
        self.tickers_ftse100 = [ticker.split('.')[0]+'.L' for ticker in self.tickers_ftse100]
        self.tickers_ftse100[self.tickers_ftse100.index('BT.L')] = 'BT-A.L' # BT Group plc (BT-A.L) on yahoo finance

        self.tickers_ftse250 = [ticker.split('.')[0]+'.L' for ticker in self.tickers_ftse250]

    def get_all_tickers(self):
        """ Get a list of all tickers in US market """
        ticker_list = self.tickers_ftse100 + \
                      self.tickers_ftse250 + \
                      self.tickers_nasdaq + \
                      self.tickers_other + \
                      self.tickers_sp500     
                      #self.tickers_dow + \
                        
        
        return set(ticker_list)    

    def get_cheap_stocks(self, tickers, threshold=0.6, start_date='2020-02-20', end_date=None, interval='1d'):
        """
        Get cheap stocks since the COVID19 pandemic. Define 'cheap' as 60% of 2020-02-20 price levels. 

        Args:
            tickers (list): A list of tickers. 
            threshold (float, optional): Threshold representing fraction of current price to 2020-02-20 price. Defaults to 0.6.
            start_date (str, optional): Start date of data. Defaults to '2020-02-20'.
            end_date (str, optional): End date of data. Defaults to None.
            interval (str, optional): Data resolution. Defaults to '1d'.

        Output is stored in self.stock_picks.
        """
        for tk in tqdm(tickers):
            tk = tk.upper()
            if '^' in tk: 
                pass
            else:
                try:    
                    my_stock = Stock(tk).get_technical_data(start_date, 
                                                              end_date,
                                                              interval)

                    current_price = my_stock['adjclose'].iloc[-1]
                    prepandemic_price = my_stock['adjclose'].iloc[0]
                    ratio = current_price / prepandemic_price
                                                                            
                    # Apply Thresehold sigma from mean
                    if ratio <= threshold: 
                        print(f'{tk} is {ratio*100:.2f} % of prepandemic price levels.')
                        self.stock_picks[tk] = Stock(tk)
                        self.tickers_high_vol.append(tk)
                except Exception as e:
                    print(e)
                    print(f'{tk} invalid.')
                    pass

    def get_high_vol_stocks(self, tickers, threshold=3, start_date=datetime.now()-timedelta(days=30),
                           end_date=None, interval='1d'):
        """ 
        Get high volume stocks which have a 3 sigma spike relative to previous 30 day average volume. 

        Args:
            tickers (list): A list of tickers. 
            threshold (int, optional): Number of sigma from mean. Defaults to 3.
            start_date (datetime, optional): Start date of data. Defaults to datetime.now()-timedelta(days=30).
            end_date (datetime, optional): End date of data. Defaults to None.
            interval (str, optional): Data resolution. Defaults to '1d'.

        Output is recorded in self.stock_picks

        """
        # Since there are over 10k stocks. Need to define some 
        
        for tk in tqdm(tickers):
            tk = tk.upper()
            if '^' in tk:
                pass
            else:
                try: 
                    my_stock = Stock(tk).get_technical_data(start_date, 
                                                              end_date,
                                                              interval)
                    prev_vol_avg = (my_stock['volume']
                                    .iloc[:-1]
                                    .mean())
                    prev_vol_std = (my_stock['volume']
                                    .iloc[:-1]
                                    .std())
                    todays_vol = my_stock['volume'].iloc[-1]
                    
                    # Thresehold sigma from mean
                    if (todays_vol - prev_vol_avg)/prev_vol_std >= threshold: 
                       # print(tk, prev_vol_std)
                        self.stock_picks[tk] = Stock(tk)
                        self.tickers_high_vol.append(tk)
                except Exception as e: 
                    print(e)
                    print(f'{tk} invalid.')
                    pass
                
                # Stop the scrape after 100 high volume stocks
                # if len(self.tickers_high_vol) > 100:
                #     break
    
    
    def is_high_vol(self, ticker, start_date=datetime.now()-timedelta(days=30),
                           end_date=None, interval='1d', threshold=3):
        """ Check if a stock is classed as high vol """
        my_stock = Stock(ticker).get_technical_data(start_date, 
                                                  end_date,
                                                  interval)
        prev_vol_avg = (my_stock['volume']
                        .iloc[:-1]
                        .mean())
        prev_vol_std = (my_stock['volume']
                        .iloc[:-1]
                        .std())
        todays_vol = my_stock['volume'].iloc[-1]
        
        # Thresehold sigma from mean
        if (todays_vol - prev_vol_avg)/prev_vol_std >= threshold: 
            return True
        
        
    def get_train_data_for_my_stocks(self):
        """ Get training data for each high volume stock in my_stocks """
        for tk in tqdm(self.stock_picks.keys()):
            #print(tk)
            s = self.stock_picks[tk]
            s.download_data()
            if s.errorCount > 0:
                continue
            s.get_train_data()
            s.preprocess_train_data()
        
        self._filter_my_stocks()
        self._combine_train_data()
            
        
    def _filter_my_stocks(self):
        """ Delete stocks without fundamental data """
        tickers_to_remove = []
        for tk in self.stock_picks.keys():
            if self.stock_picks[tk].errorCount > 0:
                tickers_to_remove.append(tk)
                
        for tk in tickers_to_remove:
            del self.stock_picks[tk]
            
        self.tickers_high_vol = list(self.stock_picks.keys())
            
    
    def _combine_train_data(self):
        for tk in self.stock_picks.keys():                
            if self.X is None:
                # -1 index to remove last row (no buy sell label because don't know tomorrow's price yet)
                self.X = self.stock_picks[tk].X[:-1] 
                self.label = self.stock_picks[tk].y[:-1]
            else:
                self.X = np.concatenate((self.X, self.stock_picks[tk].X[:-1]), axis=0)
                self.label = np.concatenate((self.label, self.stock_picks[tk].y[:-1]), axis=0)
            
        self.y = pd.Series(self.label.flatten()).astype('category').cat.codes.values


class Model:
    def __init__(self, model, stock_scanner):
        self.model = model
        self.ss = stock_scanner
        self.accuracy = None
        self.y_pred = None
        self.y_prob = None
        self.X = self.ss.X
        self.y = self.ss.y
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
    
    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                                  self.X, self.y, 
                                                  stratify=self.y, 
                                                  test_size=0.2,
                                                  shuffle=True,
                                                  random_state=42)
        
    def train(self, partial_fit=False):
        
        if self.X_train is None:
            self.train_test_split()
            
        if partial_fit:
            self.model.partial_fit(self.X_train,
                                   self.y_train,
                                   classes=np.unique(self.y))
        else:
            self.model.fit(self.X_train,
                           self.y_train)
    
    def predict(self):
        self.y_pred = self.model.predict(self.X_test)
        self.y_prob = np.max(self.model.predict_proba(self.X_test),axis=1)
    
    def get_accuracy(self):
        self.accuracy_train = self.model.score(self.X_train, self.y_train)
        self.accuracy_test = self.model.score(self.X_test, self.y_test)
        print('Train acc: ', self.accuracy_train, 'Test acc: ', self.accuracy_test)
  


def predict_today(model, my_stocks):
    """ Get predictions for today """
    my_dict = []
    for tk in tqdm(my_stocks.keys()):
        s = my_stocks[tk]
        s.pred = model.predict(s.X[-1:])
        s.prob = np.max(model.predict_proba(s.X[-1:]),axis=1)
        pred = 'Buy' if s.pred == 1 else 'Sell'
        my_dict.append({'Ticker':tk, 'Pred':pred, 'Prob':s.prob[0]})
    df = pd.DataFrame(my_dict)
    return df

def save_model(model):
    # Save to file in the current working directory
    pkl_filename = os.path.join("pickle_model.pkl")
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

def load_pkl_model(pkl_filename="pickle_model.pkl"):
    # Load from file
    pkl_filename = os.path.join(pkl_filename)
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model
