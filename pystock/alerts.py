# -*- coding: utf-8 -*-
"""
Created on Thursday June 22 13:58:19 2023

@author: I Kit Cheng

script: alerts.py
"""

import numpy as np
import os
import yfinance as yf
from signals import Signal
from indicators import Indicator
from email_notification import send_email


def alert(stock_ticker, data_period='7d', data_intvl='1h', indicator_period=4):
    """ Alert user if stock price is above upper band of Bollinger Band and send email notification. 
    """
    df = yf.download(tickers=stock_ticker, period=data_period, interval=data_intvl) 
    indicators = Indicator(df, period=indicator_period)
    signals = Signal(indicators)
    if signals.price_above_upper():
        message = f'Price of {stock_ticker} is above upper band'
        print(message)
    else:
        message = f'Price of {stock_ticker} is below upper band'
        print(message)
    
    send_email_alert(message)

def send_email_alert(message):
    subject = "Stock Alert!"
    body = message
    sender_email = "matthewkit@gmail.com"
    receiver_email = "matthewkit@gmail.com"
    smtp_server = "smtp.gmail.com"  # Gmail SMTP server
    smtp_port = 465  # SSL/TLS port
    username = os.environ.get('username', None) # Your Gmail email address
    password = os.environ.get('password', None)  # Your Gmail password or app-specific password
    send_email(subject, body, sender_email, receiver_email, smtp_server, smtp_port, username, password)

def main():
    alert('TSLA')

if __name__ == '__main__':
    main()