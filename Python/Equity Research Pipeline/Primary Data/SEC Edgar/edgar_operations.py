import pandas as pd
import requests as req
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

headers = {"User-Agent":"abc@gmail.com"}


def cik_matching_ticker(ticker, headers=headers):
    ticker = ticker.upper().replace(".","-")
    ticker_json = req.get('https://www.sec.gov/files/company_tickers.json', headers=headers).json()
    for company in ticker_json.values():
        if company["ticker"] == ticker:
            cik = str(company["cik_str"]).zfill(10)
            return cik
    raise ValueError(f"Ticker {ticker} not found in SEC database")

def get_submission_data(ticker, headers=headers, only_filings_df=False):
    cik = cik_matching_ticker(ticker)
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    company_json = req.get(url,headers=headers).json()
    if only_filings_df:
        return pd.DataFrame(company_json['filings']['recent'])
    return company_json

def get_filtered_filings(ticker, headers=headers, ten_k=True,just_accession_numbers=False):
    company_filings_df = get_submission_data(ticker,headers,only_filings_df=True)
    if ten_k:
        df = company_filings_df[company_filings_df['form'] == "10-K"]
    else:
        df = company_filings_df[company_filings_df['form'] == "10-Q"]
    if just_accession_numbers:
        df = df.set_index('reportDate')
        accession_df = df['accessionNumber']
        return accession_df
    else:
        return df
    
def get_financial_data(ticker,headers=headers):
    cik = cik_matching_ticker(ticker)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    data = req.get(url,headers=headers).json()
    return data

def facts_df(ticker, headers=headers):
    data = get_financial_data(ticker)
    us_gaap_data = data['facts']['us-gaap']
    df_data = []
    for fact, details in us_gaap_data.items():
        for unit in details['units']:
            for item in details['units'][unit]:
                row = item.copy()
                row["fact"] = fact
                df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df["end"] = pd.to_datetime(df["end"])
    df["start"] = pd.to_datetime(df["start"])
    df = df.drop_duplicates(subset=["fact","end","val"])
    df.set_index("end",inplace=True)
    labels_dict = {fact: details["label"] for fact, details in us_gaap_data.items()}
    return df, labels_dict

def time_series_comparison(df,target_line,headers=headers):

    time_series_financials = df[df['fact']==target_line][['fact','val']]

    fig,ax = plt.subplots(figsize=(15,6))

    ax.plot(time_series_financials['val'],color="red",alpha=0.85,lw=1.5,label=f"{target_line}")
    ax.set_title(f"{target_line} change from {pd.to_datetime(time_series_financials.index[0]).year} to {pd.to_datetime(time_series_financials.index[-1]).year}")
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{target_line}')
    ax.legend()
    plt.show()

def time_series_comparison_v2(df, target_lines, headers=headers, 
                          normalize=False, figsize=(15, 8), title=None):

    if isinstance(target_lines, str):
        target_lines = [target_lines]
    
    available_metrics = df.index.tolist()
    missing_metrics = [m for m in target_lines if m not in available_metrics]
    
    if missing_metrics:
        print(f"Warning: Metrics not found: {missing_metrics}")
        target_lines = [m for m in target_lines if m in available_metrics]
    
    if not target_lines:
        print(" rror: No valid metrics to plot")
        return None
    
    time_series_data = {}
    for metric in target_lines:
        series = df.loc[metric].dropna()
        if not series.empty:
            time_series_data[metric] = series
    
    if not time_series_data:
        print("Error: No valid data available")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, (metric, series) in enumerate(time_series_data.items()):
        color = colors[i % len(colors)]
        
        if normalize and not series.empty:
            normalized = (series / series.iloc[0]) * 100
            ax.plot(normalized.index, normalized.values, color=color, 
                   alpha=0.85, lw=1.8, label=f"{metric} (norm)", 
                   marker='o', markersize=5)
        else:
            ax.plot(series.index, series.values, color=color, 
                   alpha=0.85, lw=1.8, label=metric, 
                   marker='o', markersize=5)
    
    if title is None:
        metrics_str = ', '.join(target_lines[:3])
        if len(target_lines) > 3:
            metrics_str += f"... ({len(target_lines)} total)"
        title = f"Financial Metrics: {metrics_str}"
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Value' + (' (Normalized to 100)' if normalize else ''), fontsize=12)
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.show()


def annual_facts(ticker, headers=headers):
    accession_nums = get_filtered_filings(
        ticker,ten_k=True,just_accession_numbers=True
    )
    df, label_dict = facts_df(ticker,headers)
    ten_k = df[df['accn'].isin(accession_nums)]
    ten_k = ten_k[ten_k.index.isin(accession_nums.index)]
    pivot = ten_k.pivot_table(values="val",columns="fact",index="end")
    pivot.rename(columns = label_dict,inplace=True)
    return pivot.T

def quarterly_facts(ticker, headers=headers):
    accession_nums = get_filtered_filings(
        ticker,ten_k=False,just_accession_numbers=True
    )
    df, label_dict = facts_df(ticker,headers)
    ten_q = df[df['accn'].isin(accession_nums)]
    ten_q = ten_q[ten_q.index.isin(accession_nums.index)]
    pivot = ten_q.pivot_table(values="val",columns="fact",index="end")
    pivot.rename(columns = label_dict,inplace=True)
    return pivot.T


def all_quarterly_facts(ticker, headers=headers):

    def is_balance_sheet_item(fact_name):
        balance_sheet_keywords = [
            'Asset', 'Liability', 'Equity', 'Cash', 'Receivable', 'Payable',
            'Inventory', 'Property', 'Equipment', 'Goodwill', 'Intangible',
            'Investment', 'Debt', 'Capital', 'Retained', 'Stockholders',
            'SharesOutstanding', 'CommonStock', 'PreferredStock', 'Treasury',
            'AccumulatedDepreciation', 'Allowance', 'DeferredTax', 'Accrued',
            'Prepaid', 'Warrant', 'Derivative', 'Lease', 'Deferred'
        ]
        fact_upper = str(fact_name).upper()
        return any(kw.upper() in fact_upper for kw in balance_sheet_keywords)
    
    accession_nums_q = get_filtered_filings(ticker, ten_k=False, just_accession_numbers=True)
    accession_nums_k = get_filtered_filings(ticker, ten_k=True, just_accession_numbers=True)
    
    df, label_dict = facts_df(ticker, headers)
    
    ten_q = df[df['accn'].isin(accession_nums_q) & df.index.isin(accession_nums_q.index)]
    pivot_q = ten_q.pivot_table(values="val", columns="fact", index="end", aggfunc='first')
    
    ten_k = df[df['accn'].isin(accession_nums_k) & df.index.isin(accession_nums_k.index)]
    pivot_k = ten_k.pivot_table(values="val", columns="fact", index="end", aggfunc='first')
    
    all_quarters = pivot_q.copy()
    
    for year_end in pivot_k.index:
        q_dates = pivot_q.index[pivot_q.index < year_end]
        
        if len(q_dates) >= 3:
            q_dates = sorted(q_dates)[-3:]  
            
            for fact in pivot_k.columns:
                annual_val = pivot_k.loc[year_end, fact]
                
                if pd.notna(annual_val):
                    fact_label = label_dict.get(fact, fact)
                    
                    if is_balance_sheet_item(fact_label):
                        all_quarters.loc[year_end, fact] = annual_val
                    else:
                        if fact in pivot_q.columns:
                            q123_sum = pivot_q.loc[q_dates, fact].sum()
                            if pd.notna(q123_sum) and q123_sum != 0:
                                all_quarters.loc[year_end, fact] = annual_val - q123_sum
                            else:
                                all_quarters.loc[year_end, fact] = annual_val
                        else:
                            all_quarters.loc[year_end, fact] = annual_val
    
    all_quarters = all_quarters.sort_index()
    all_quarters.rename(columns=label_dict, inplace=True)
    
    return all_quarters.T
