
## Overview

This R script performs comprehensive time series analysis on electricity generation data across multiple fuel sources. The analysis includes data preprocessing, exploratory data analysis, statistical modeling, forecasting, and advanced analytics to understand patterns and predict future electricity generation.

## Dataset

**File**: `Net_generation_for_all_sectors.csv`
- **Format**: Monthly electricity generation data
- **Columns**: Date, All_Fuels, Coal, Petroleum_Liquids, Petroleum_Coke, Natural_Gas, Other_Gases, Nuclear, Conventional_Hydro, Other_Renewables, Pumped_Storage, Other
- **Unit**: MWh (Megawatt hours)

## Dependencies

### Required R Packages
```r
# Time Series Analysis
forecast, TSstudio, tseries, urca, vars, prophet, fable, feasts, tsibble, fabletools, modeltime, timetk

# Data Manipulation & Visualization
tidyverse, lubridate, plotly, ggplot2, gridExtra, corrplot, scales, RColorBrewer

# Advanced Analytics
bcputility, changepoint, MTS, tsDyn, MARSS, fracdiff

# Machine Learning
randomForest, xgboost, caret, e1071, glmnet

# Statistical Tests
car, nortest, moments

# Reporting
knitr, kableExtra
```

## Analysis Components

### 1. Data Loading and Preprocessing
- **Input**: Raw CSV file with electricity generation data
- **Output**: Clean time series dataset
- **Features**:
  - Missing value imputation using linear approximation
  - Date parsing and standardization
  - Column name cleaning and standardization

### 2. Exploratory Data Analysis (EDA)
- **Time Series Plots**: Individual fuel source trends over time
- **Correlation Matrix**: Relationships between different fuel sources
- **Seasonal Decomposition**: STL and classical decomposition
- **Temporal Patterns**: Monthly and yearly distribution analysis

#### Generated Visualizations:
- Time series plots for each fuel source
- Correlation heatmap
- STL decomposition plots
- Monthly/yearly boxplots
- Seasonal pattern analysis

### 3. Stationarity Analysis
- **Tests Performed**:
  - Augmented Dickey-Fuller (ADF) Test
  - Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test
  - Phillips-Perron (PP) Test
  - Unit Root Test
- **Variables Tested**: All_Fuels, Coal, Natural_Gas, Nuclear

### 4. Data Transformations
- **Original Series**: Raw time series data
- **Log Transformation**: Log(x + 1) transformation
- **First Differencing**: Removes trend component
- **Seasonal Differencing**: Removes seasonal component (lag = 12)
- **Box-Cox Transformation**: Optimal lambda parameter estimation
- **Log + Differencing**: Combined transformation

### 5. ACF/PACF Analysis
- **Autocorrelation Function (ACF)**: Identifies seasonal patterns
- **Partial Autocorrelation Function (PACF)**: Determines AR order
- **Differenced Series Analysis**: Post-transformation correlation structure

### 6. Univariate Time Series Models

#### ARIMA Models
- **Auto ARIMA**: Automated model selection
- **Custom ARIMA Models**:
  - ARIMA(1,1,1)(1,1,1)
  - ARIMA(2,1,2)(2,1,2)
  - ARIMA(0,1,1)(0,1,1)

#### Other Models
- **ETS Model**: Exponential Smoothing State Space
- **Structural Time Series**: Basic Structural Model (BSM)

### 7. Multivariate Analysis (VAR)
- **Variables**: All_Fuels, Coal, Natural_Gas, Nuclear, Other_Renewables
- **Cointegration Testing**: Johansen cointegration test
- **Lag Selection**: AIC-based optimal lag determination
- **Granger Causality**: Directional causality testing
- **Impulse Response Functions**: Shock propagation analysis

#### VAR Diagnostics:
- Serial correlation test
- ARCH effects test
- Multivariate normality test

### 8. Prophet Model
- **Features**:
  - Yearly seasonality
  - Multiplicative seasonality mode
  - Automatic trend detection
- **Components Analysis**: Trend, seasonal, and residual decomposition

### 9. Machine Learning Models

#### Feature Engineering
- **Lag Features**: 1-12 period lags
- **Moving Averages**: 3, 6, 12-period moving averages
- **Seasonal Lags**: 12 and 24-period seasonal lags
- **Temporal Features**: Trend, month, quarter indicators

#### Models Implemented
- **Random Forest**: 500 trees with importance analysis
- **XGBoost**: Gradient boosting with feature importance
- **Support Vector Regression (SVR)**: Radial basis function kernel

### 10. Model Evaluation and Comparison

#### Metrics Used
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **MAPE**: Mean Absolute Percentage Error

#### Models Compared
- ARIMA
- ETS
- Prophet
- Random Forest
- XGBoost
- SVR

### 11. Advanced Analytics

#### Change Point Detection
- **Methods**: PELT (Pruned Exact Linear Time)
- **Types**: Mean change, variance change, mean-variance change

#### Regime Switching Models
- **Markov Switching Models**: Two-regime model
- **State Probability Analysis**: Regime transition probabilities

#### Spectral Analysis
- **Periodogram**: Frequency domain analysis
- **Dominant Frequencies**: Peak detection in spectrum

### 12. Seasonal and Trend Analysis

#### Seasonal Patterns
- **Winter**: December, January, February
- **Spring**: March, April, May
- **Summer**: June, July, August
- **Fall**: September, October, November

#### Yearly Trends
- Total generation trends
- Fuel mix evolution over time
- Market share analysis by fuel type

#### Peak Analysis
- Top 10 highest generation periods
- Peak demand identification
- Seasonal peak patterns

#### Volatility Analysis
- Coefficient of variation by fuel source
- Risk assessment
- Stability ranking

## Outputs

### CSV Files Generated
1. **model_performance_comparison.csv**: Model evaluation metrics
2. **electricity_generation_forecast.csv**: Future predictions
3. **yearly_generation_trends.csv**: Annual trend analysis

### Visualizations Generated
1. Time series plots for all fuel sources
2. Correlation matrix heatmap
3. STL decomposition plots
4. ACF/PACF plots
5. Forecast comparison plots
6. Variable importance plots
7. Change point detection plots
8. Spectral analysis plots
9. Model performance comparison charts
10. Dashboard summary plots

### Statistical Outputs
1. Stationarity test results
2. Model diagnostic tests
3. Cointegration test results
4. Granger causality test results
5. Change point locations
6. Dominant frequency periods

## Production-Ready Functions

### `electricity_forecast(data, horizon, model_type)`
- **Purpose**: Generate forecasts for electricity generation
- **Parameters**:
  - `data`: Input dataset with Date and All_Fuels columns
  - `horizon`: Forecast horizon (default: 12 months)
  - `model_type`: "arima", "ets", "prophet", or "ensemble"
- **Returns**: Dataframe with forecast dates and predictions

### `deploy_forecast(new_data, horizon)`
- **Purpose**: Production deployment function
- **Features**:
  - Input validation
  - Confidence intervals
  - Metadata tracking
  - Version control

## Key Findings

### Seasonal Patterns
- Strong seasonal variations with peak demand in summer/winter months
- Monthly patterns consistent across years
- Quarterly effects significant for capacity planning

### Fuel Mix Evolution
- Transition from coal to natural gas and renewables
- Nuclear generation remains stable
- Renewable energy growth trend

### Volatility Analysis
- Natural gas: Highest volatility
- Nuclear: Most stable generation
- Coal: Declining with medium volatility

### Model Performance
- Ensemble methods provide best accuracy
- Machine learning models excel in complex pattern recognition
- Traditional time series models good for interpretability

### Structural Changes
- Multiple regime changes detected in historical data
- Significant policy and market-driven shifts
- Change points align with major energy policy changes

## Recommendations

### Capacity Planning
- Plan expansions based on seasonal demand patterns
- Account for peak demand periods
- Consider regional variations

### Fuel Diversification
- Continue shift away from coal
- Increase renewable energy capacity
- Maintain nuclear baseload capability

### Grid Stability
- Increase energy storage capacity
- Improve grid flexibility
- Plan for renewable intermittency

### Forecasting Strategy
- Implement ensemble forecasting methods
- Regular model retraining and validation
- Monitor forecast accuracy continuously

### Risk Management
- Hedge against natural gas price volatility
- Plan for extreme weather events
- Diversify generation portfolio

## Usage

1. **Setup**: Install required packages using the provided installation function
2. **Data**: Place `Net_generation_for_all_sectors.csv` in the working directory
3. **Execution**: Run the script sections sequentially
4. **Customization**: Modify parameters for specific analysis needs
5. **Output**: Review generated files and visualizations

## Model Selection Guide

- **For accuracy**: Use ensemble approach
- **For interpretability**: Use ARIMA or ETS
- **For complex patterns**: Use XGBoost or Random Forest
- **For business reporting**: Use Prophet
- **For policy analysis**: Use VAR models

## Performance Benchmarks

Expected model performance (typical ranges):
- **RMSE**: 50,000 - 150,000 MWh
- **MAPE**: 3% - 8%
- **MAE**: 40,000 - 120,000 MWh

## Version Information

- **Script Version**: 1.0
- **R Version Required**: >= 4.0.0
- **Last Updated**: [DATE]