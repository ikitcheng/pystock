
# PyStock

`PyStock` is a python package that scans stocks and predicts prices. 

## Installation

Browse to the directory where this file lives, and run:
```bash
pip install .
```
This command will download any dependencies we have.

## Stock Scanner

* Find abnormally high volume stock
* Get technical data: OHLC, pct_change, volatility
* Quantify fundamental data: EPS, Market cap, Revenue growth, etc. (used to filter high volume stocks)
	* Is company growing?
	* Is it in an innovative sector? 
* Media attention over some reference period: positive/ negative catalysts? 
* Create risk metric for the stock (0 low to 1 high- maybe with sigmoid function?)
	- The more days it goes green, the higher the risk.
	- The number of days elapsed is from the day the volume exploded.
	- Should be weighted somehow by news. 
* Predict probability that price will go up on next market open. 


## Todo
* add ma50, ma200
* Filter high vol stocks based on fundamental data
* Attention model for timeseries forecasting