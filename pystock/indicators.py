# -*- coding: utf-8 -*-
"""
Created on Thursday June 22 13:58:19 2023

@author: I Kit Cheng

script: indicators.py
"""
import numpy as np

class Indicator():
    def __init__(self, df, period):
        """ Indicators class constructor.

        Args:
            df (pd.DataFrame): stock dataframe containing ohlcv.
        """
        self.df = df.copy()
        self.period = period
        self.get_indicators()

    def get_indicators(self):
        self.df[f'SMA{self.period}'] = self.sma(self.period)
        self.df[f'STD{self.period}'] = self.std(self.period)
        self.df['Upper'] = self.upper()
        self.df['Lower'] = self.lower()
        self.df[f'EWMA{self.period}'] = self.ewma(self.period)
        self.df[f'Momentum{self.period}'] = self.momentum(self.period)
        self.df['Log'] = self.log()

    # Create a method for each stock indicator
    def sma(self, period=20):
        """Calculate simple moving average of stock
        """
        return self.df['Adj Close'].rolling(window=period).mean()
    
    def std(self, period=20):
        """Calculate standard deviation of stock
        """
        return self.df['Adj Close'].rolling(window=period).std()
    
    def upper(self):
        """Calculate upper band of Bollinger Band (2 std above sma)"""
        return self.df[f'SMA{self.period}'] + (self.df[f'STD{self.period}'] * 2)
    
    def lower(self):
        """Calculate lower band of Bollinger Band (2 std below sma)"""
        return self.df[f'SMA{self.period}'] - (self.df[f'STD{self.period}'] * 2)
    
    def ewma(self, period=20):
        """ Calculate exponentially weighted moving average """
        return self.df['Adj Close'].ewm(span=period).mean()
    
    def momentum(self, period):
        """ Calculate momentum """
        # Calculate the price change
        price_change = self.df['Adj Close'].diff(periods=period)

        # Calculate the momentum (%change since `period` ago)
        momentum = price_change / self.df['Adj Close'].shift(periods=period) * 100

        return momentum
    
    def log(self):
        """ Calculate log of stock price """
        self.df['Log'] = np.log(self.df['Adj Close'])
        return self.df['Log']