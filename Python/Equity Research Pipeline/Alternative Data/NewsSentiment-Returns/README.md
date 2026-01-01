## Google News Sentiment & Stock Return Prediction
A quantitative finance pipeline that leverages alternative data to predict stock returns. It fetches news via pygooglenews, quantifies sentiment using a Transformer model (DistilBERT), and builds a regularized Lasso Regression model to forecast price movements.

## 🚀 Features
Automated Scraping: Fetches historical news for any ticker.

NLP Sentiment Scoring: Classifies headlines as Positive/Negative with confidence scores.

Polarity Score: Normalizes sentiment daily: (Pos - Neg) / Total.

Lasso Regression: Predicts future returns using L1 regularization and TimeSeriesSplit CV.

Metric: Evaluates performance using Information Coefficient (IC).

## 🛠️ Quick Usage
python
### 1. Fetch & Analyze Sentiment
df = sentiment_analysis("Tesla", "2025-06-30", "2025-12-01", "label")

### 2. Process Time Series
ts_data = time_series_sentiment(df, "Tesla", sma=5)

### 3. Merge with Market Data (TSLA)
analysis_df = sentiment_returns_analysis(ts_data, "TSLA")

### 4. Train Model (Predict 3 days ahead)
model = regression_model_returns_prediction(analysis_df, n_splits=2, x_days_later=3)
📊 Methodology
Extract: Get headlines for target asset.

Transform: Compute daily sentiment polarity (-1 to +1).

Model: Train Lasso Regressor on sentiment vs. future returns.

Evaluate: Check IC scores to validate predictive power.

## 📂 Structure
sentiment_fetch.py: News scraping.

gn_analytics.py: NLP & Visualization.

gn_lasso_model.py: ML Pipeline (GridSearchCV, Lasso).

For educational/research use only.
