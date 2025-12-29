import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ticker1 = str(input("Enter the 1st stock selection: "))
ticker2 = str(input("Enter the 2nd stock selection: "))
ticker3 = str(input("Enter the 3rd stock selection: "))
ticker4 = str(input("Enter the 4th stock selection: "))
ticker5 = str(input("Enter the 5th stock selection: "))
ticker6 = str(input("Enter the 6th stock selection: "))
ticker7 = str(input("Enter the 7th stock selection: "))
ticker8 = str(input("Enter the 8th stock selection: "))
ticker9 = str(input("Enter the 9th stock selection: "))
ticker10 = str(input("Enter the 10th stock selection: "))

weight1 = float(input("Input the weight for the 1st stock: "))
weight2 = float(input("Input the weight for the 2nd stock: "))
weight3 = float(input("Input the weight for the 3rd stock: "))
weight4 = float(input("Input the weight for the 4th stock: "))
weight5 = float(input("Input the weight for the 5th stock: "))
weight6 = float(input("Input the weight for the 6th stock: "))
weight7 = float(input("Input the weight for the 7th stock: "))
weight8 = float(input("Input the weight for the 8th stock: "))
weight9 = float(input("Input the weight for the 9th stock: "))
weight10 = float(input("Input the weight for the 10th stock: "))

rfr = float(input("Enter the risk-free rate: "))/100
ticker_list = [ticker1,ticker2,ticker3,ticker4,ticker5,ticker6,ticker7,ticker8,ticker9,ticker10]

stock1 = yf.Ticker(ticker1).history(period = "5Y", interval = "1D").Close.pct_change().dropna()
stock2 = yf.Ticker(ticker2).history(period = "5Y", interval = "1D").Close.pct_change().dropna()
stock3 = yf.Ticker(ticker3).history(period = "5Y", interval = "1D").Close.pct_change().dropna()
stock4 = yf.Ticker(ticker4).history(period = "5Y", interval = "1D").Close.pct_change().dropna()
stock5 = yf.Ticker(ticker5).history(period = "5Y", interval = "1D").Close.pct_change().dropna()
stock6 = yf.Ticker(ticker6).history(period = "5Y", interval = "1D").Close.pct_change().dropna()
stock7 = yf.Ticker(ticker7).history(period = "5Y", interval = "1D").Close.pct_change().dropna()
stock8 = yf.Ticker(ticker8).history(period = "5Y", interval = "1D").Close.pct_change().dropna()
stock9 = yf.Ticker(ticker9).history(period = "5Y", interval = "1D").Close.pct_change().dropna()
stock10 = yf.Ticker(ticker10).history(period = "5Y", interval = "1D").Close.pct_change().dropna()

stock1.index = pd.to_datetime(stock1.index).date
stock2.index = pd.to_datetime(stock2.index).date
stock3.index = pd.to_datetime(stock3.index).date
stock4.index = pd.to_datetime(stock4.index).date
stock5.index = pd.to_datetime(stock5.index).date
stock6.index = pd.to_datetime(stock6.index).date
stock7.index = pd.to_datetime(stock7.index).date
stock8.index = pd.to_datetime(stock8.index).date
stock9.index = pd.to_datetime(stock9.index).date
stock10.index = pd.to_datetime(stock10.index).date

mdata = pd.DataFrame({"Stock1":stock1, "Stock2":stock2, "Stock3":stock3, "Stock4":stock4, "Stock5":stock5, "Stock6":stock6,"Stock7":stock7, "Stock8":stock8,"Stock9":stock9,"Stock10":stock10})
mdata = mdata.dropna()

for i in range(1,11):
  mdata[f"Stock{i} N-Price"] = 100*(1+mdata[f"Stock{i}"]).cumprod()

color_list = ["blue", "red", "green", "orange", "purple", "cyan", "magenta", "yellow", "black", "brown"]

correlation = mdata[["Stock1","Stock2","Stock3","Stock4","Stock5","Stock6","Stock7","Stock8","Stock9","Stock10"]].corr()
sns.heatmap(correlation, annot = True)

p_return = 0

for i in range(1,11):
  p_return += mdata[f"Stock{i}"].mean()

p_return = p_return/10

weights = np.array([weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, weight9, weight10])


returns_array = mdata[[f"Stock{i}" for i in range(1, 11)]].values


cov_matrix = np.cov(returns_array.T)


portfolio_variance = weights.T @ cov_matrix @ weights


portfolio_volatility = np.sqrt(portfolio_variance)/100

annual_volatility = portfolio_volatility * np.sqrt(252)

annual_return = p_return*252
sharpe_ratio = (annual_return-rfr)/annual_volatility

print(f'annual volatility: {annual_volatility}')
print(f'annual return: {annual_return}')
print(f'sharpe ratio: {sharpe_ratio}')
