from pytrends.request import TrendReq
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import yfinance as yf

use_colors = [
    "#000000",  
    "#1800B2",  
    "#FF9900FF",  
    "#004633",  
    "#56B4E9", 
    "#9D5D2C",  
    "#79CCA9",  
    "#F0E442"
]

def time_series_interest(query_list, region="", 
                         start_date="2010-01-01", 
                         end_date=str(dt.date.today())):
    
    plt.style.use("seaborn-v0_8-whitegrid")

    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload(kw_list=query_list, timeframe = start_date + " " + 
                           end_date, geo=region)
    interest_df = pytrends.interest_over_time()
    
    fig, ax = plt.subplots(figsize=(15,8))
    ax.set_prop_cycle(cycler(color=use_colors))
    for col in interest_df.columns[:-1]:       
        ax.plot(interest_df.index, interest_df[col], label=col)

    ax.set_title(f"Time Series Plot for {[f"{query_list[x]}, " if x != len(query_list)-1 else query_list[x] for x in range(len(query_list))]}",
                 fontsize=14)
    ax.set_ylabel("Google Search Interest")
    ax.set_xlabel("Date")
    ax.legend(fontsize=14)
    plt.show()
    return interest_df

def interest_returns_combination(interest_df,ticker:str):

    plt.style.use("classic")

    returns_df = yf.download(tickers=ticker,
                             start=interest_df.index[0],
                             end=interest_df.index[-1],interval="1mo")["Close"].pct_change()
    combined_df = pd.concat([returns_df,interest_df.iloc[:,:-1]],axis=1,join="inner").dropna()

    fig,ax = plt.subplots(figsize=(15,8))
    ax2 = ax.twinx()
    ax.plot(combined_df[combined_df.columns[0]],color='red',label="returns",alpha=0.7,lw=2)

    for i in range(1,len(combined_df.columns)):
        ax2.plot(combined_df[combined_df.columns[i]],color=use_colors[i],
                 label=f"Google Search Interest: {combined_df.columns[i]}",
                 lw=1.2)

    ax.set_ylabel("Returns")
    ax2.set_ylabel("Interest")
    ax.set_xlabel("Date")
    plt.title(f"Google Search Interest & Monthly Stock returns for {interest_df.columns[0]}")
    fig.legend(loc="upper right")
    plt.show()

    return combined_df



