import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ticker = input("Enter the ticker: ")
stock = yf.Ticker(str(ticker))
data = stock.history(period = "1d", interval = "1m")

data["Return"] = (data.Close/data.Close.shift(1)-1).dropna()
data["Move"] = data.Close - data.Open

#Overall Descriptive Data
volatility = (data.Return.std())*100
total_return = round(((data.Close[-1]/data.Open[0])-1)*100,3)

# MA Columns Making
data["SMA-20M"] = data.Close.rolling(window = 20).mean()
data["SMA-50M"] = data.Close.rolling(window=50).mean()

#RSI Computation
RSI_valuelist = []
for i in range(len(data)):
  if i<14:
    RSI_valuelist.append(np.nan)
  else:
    rolling_window = data["Move"].iloc[i-14:i].tolist()
    positive_total = 0
    negative_total = 0
    for j in rolling_window:
      if j > 0:
        positive_total = positive_total + j
      elif j < 0:
        negative_total = negative_total - j
    if negative_total !=0:
     RS = positive_total/negative_total
    else:
      RS.np.inf
    RSI = 100 - (100/(1+RS))
    RSI_valuelist.append(RSI)

data["RSI"] = RSI_valuelist

fig, ax1 = plt.subplots()

ax1.plot(data.Close, color = "Black")
ax1.plot(data["SMA-20M"],color = "Blue")
ax1.plot(data["SMA-50M"], color = "Green")

ax2 = ax1.twinx()

ax2.plot(data["RSI"],color = "Red")

plt.title(f"Technical Indicators of {ticker}")
plt.show()
