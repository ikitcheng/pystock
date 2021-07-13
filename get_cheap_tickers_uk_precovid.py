from pystock.stock_scanner import Stock, StockScanner, Model, save_model
import pickle
import os
from datetime import datetime

def get_cheap_tickers_uk_precovid(threshold=0.6):
    """
    Get cheap uk stock tickers pre-covid levels
    Args:
        threshold (float, optional): Threshold representing fraction of current price to 2020-02-20 price. Defaults to 0.6.
    
    Output:
        my_stock_picks (dict) : Dictionary of stock tickers and pystock.stock_scanner.Stock objects.
    """
    ss = StockScanner()
    ss.get_cheap_stocks(ss.tickers_ftse100,threshold=threshold)
    cheap_ftse100 = ss.stock_picks
    ss.get_cheap_stocks(ss.tickers_ftse250)
    cheap_ftse250 = ss.stock_picks
    my_stock_picks = {**cheap_ftse100, **cheap_ftse250}
    return my_stock_picks

if __name__ == '__main__':
    my_stock_picks = get_cheap_tickers_uk_precovid(threshold=0.6)
    fname = f"./output/{datetime.now().date()}-cheap_uk_stocks.pkl"
    if not os.path.exists('output'):
        os.makedirs('output')
    with open(fname, 'wb') as f:
        pickle.dump(my_stock_picks, f)

    
