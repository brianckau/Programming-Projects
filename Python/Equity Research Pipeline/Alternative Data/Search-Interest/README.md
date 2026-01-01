# Google Search Interest - Asset Return Model
A quantitative research tool that uses Google Trends search volume to predict stock market returns. It automates the fetching of search data and uses Lasso Regression to find predictive signals.

## 🚀 Features
Automated Scraping: Fetches search volume for keywords (e.g., "Unemployment Benefit") using pytrends.

Lasso Regression: Predicts asset returns using L1-regularized linear models to filter noise.

Robust Validation: Uses TimeSeriesSplit and GridSearchCV to tune hyperparameters without look-ahead bias.

Lag Analysis: Tests different time lags (predicting 1, 2, ... 10 months ahead) to find the best forecasting window.

Metric: Evaluates models using the Information Coefficient (IC).

## 🛠️ Usage
### 1. Fetch Search Data
df = time_series_interest(["Unemployment Benefit"], start_date="2010-01-01", region="US")

### 2. Train Model
model = lasso_regression_model_for_prediction(df, n_splits=2, return_xmonthlater=0)

### 3. Check Performance
print(f"Model IC: {model.best_score_}")


##📊 Methodology
Extract: Get keyword volume history.

Model: Train Lasso Regressor on volume vs. asset returns (^GSPC).

Optimize: Find the best alpha and time lag.

Evaluate: Classify signal strength (Weak vs. Useful).
