# -*- coding: utf-8 -*-
"""
Created on Thursday June 22 13:58:19 2023

@author: I Kit Cheng

script: signals.py
"""
import numpy as np

class Signal():
    def __init__(self, indicators):
        """ Signal class constructor.

        Args:
            indicators (class): Indicator class of a specific stock.
        """

        self.indicators = indicators
    
    def price_above_upper(self):
        if self.indicators.df['Adj Close'][-1] > self.indicators.df['Upper'][-1]:
            return True
        else:
            return False
    
