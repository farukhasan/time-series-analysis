# Comprehensive Time Series Analysis of Electricity Generation Data

## Project Overview

This project analyzes U.S. electricity generation patterns using 24 years of monthly data from January 2001 to April 2025. The analysis combines traditional time series methods with modern machine learning techniques to understand energy trends and forecast future electricity generation across different fuel sources.

The dataset contains 292 monthly observations covering 11 different energy sources including coal, natural gas, nuclear, hydroelectric, and renewable energy. This comprehensive analysis reveals significant insights about America's evolving energy landscape and provides accurate forecasting capabilities for energy sector planning.

## Data Analysis and Methodology

### Initial Data Exploration

The analysis begins with extensive exploratory data analysis to understand the underlying patterns in electricity generation. The time series exhibits strong seasonal patterns with peak demand typically occurring during summer and winter months due to increased cooling and heating needs.

![Trend Analysis](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/Trend.png?raw=true)

Long-term trend analysis reveals consistent growth in total electricity generation over the 24-year period, with notable shifts in the energy mix composition. The data shows a clear transition from coal-dependent generation toward natural gas and renewable sources, reflecting both market dynamics and policy changes in the energy sector.

### Seasonal and Temporal Patterns

Understanding seasonal variations is crucial for energy planning and grid management. The analysis reveals distinct monthly and quarterly patterns that align with known demand cycles in the electricity sector.

![Monthly and Yearly Distribution](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/monthly_yearly_distribution.png?raw=true)

The seasonal analysis shows predictable patterns with higher generation during summer months (June-August) and winter months (December-February), corresponding to peak air conditioning and heating demands. These patterns are consistent across years but show some variation in magnitude, reflecting economic growth and population changes.

### Time Series Decomposition

Decomposition analysis separates the time series into trend, seasonal, and residual components, providing insight into the underlying structure of electricity generation patterns.

![STL Decomposition](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/Decomposition_STL_1.png?raw=true)

The STL (Seasonal and Trend decomposition using Loess) method effectively captures the seasonal patterns while revealing the long-term trend component. This decomposition shows that seasonal effects account for significant variation in electricity generation, with the trend component showing steady growth over the analysis period.

![Classical Decomposition](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/decomposition_2.png?raw=true)

Classical decomposition provides an alternative view of the time series structure, confirming the presence of strong seasonal patterns and an upward trend in total generation. The residual component appears relatively stable, indicating that the model captures most of the systematic variation in the data.

### Data Transformations and Preprocessing

Various transformation techniques were applied to prepare the data for modeling and analysis. These transformations help stabilize variance and improve model performance.

![Data Transformations](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/transformations.png?raw=true)

The transformation analysis includes logarithmic transformations, differencing, and Box-Cox transformations. First-order differencing effectively removes the trend component, while seasonal differencing addresses the seasonal patterns. The Box-Cox transformation helps normalize the data distribution and stabilize variance across different periods.

### Correlation Structure Analysis

Understanding relationships between different energy sources is essential for comprehensive energy analysis and forecasting.

![Correlation Matrix](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/Correlation_matrix.png?raw=true)

The correlation analysis reveals complex relationships between different electricity generation sources. The heatmap shows that Pumped Storage has strong positive correlations with several sources, indicating its role in grid balancing. Natural Gas shows moderate correlations with other sources, reflecting its flexible generation capabilities. Coal and Other Renewables display notable negative correlations with some sources, suggesting substitution effects in the energy mix. Nuclear power exhibits relatively moderate correlations, consistent with its baseload generation role. The matrix reveals the interconnected nature of the electricity generation portfolio and how different sources complement or substitute for each other.

### Autocorrelation Analysis

Autocorrelation analysis helps identify the appropriate model structure for time series forecasting.

![ACF and PACF Analysis](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/ACF_PACF.png?raw=true)

The autocorrelation function (ACF) and partial autocorrelation function (PACF) plots reveal the temporal dependencies in the data. The ACF shows strong seasonal patterns with significant correlations at 12-month lags, while the PACF helps determine the appropriate order for autoregressive models. These patterns inform the selection of ARIMA model parameters.

### Change Point Detection

Identifying structural changes in the time series helps understand major shifts in electricity generation patterns.

![Change Point Analysis](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/Change_point_detction.png?raw=true)

Change point analysis identifies several significant structural breaks in the electricity generation time series. These breakpoints often correspond to major policy changes, economic events, or technological shifts in the energy sector. The analysis reveals multiple regime changes, particularly around 2008-2009 (corresponding to the financial crisis) and 2012-2014 (reflecting the shale gas revolution).

### Regime Switching Analysis

Regime switching models capture different market conditions and operational states in electricity generation.

![Regime 1 Probabilities](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/Regime_1.png?raw=true)

The first regime analysis shows the probability distribution and time series behavior during normal operational periods. The chart displays the differenced time series (ts_diff) overlaid with smooth probabilities, where the gray shaded areas indicate periods when Regime 1 is active. The bottom panel shows the regime probabilities as vertical bars. This regime appears to dominate most of the time series, representing periods of stable electricity generation patterns with typical seasonal variations and manageable volatility levels.

![Regime 2 Probabilities](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/Regime_2.png?raw=true)

The second regime captures periods of elevated volatility or structural change in generation patterns. These periods often correspond to economic disruptions, extreme weather events, or major policy changes affecting the energy sector.

### Spectral Analysis

Frequency domain analysis provides additional insights into cyclical patterns in electricity generation.

![Periodogram](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/periodogram.png?raw=true)

The periodogram reveals the dominant frequencies in the electricity generation time series. The analysis confirms strong annual cycles (12-month periods) and identifies other significant frequency components. This spectral analysis supports the seasonal modeling approach and helps validate the decomposition results.

## Multivariate Analysis

### Vector Autoregression (VAR) Model

The VAR model analyzes interdependencies between different energy sources, providing insights into how changes in one source affect others.

![All Fuels Impulse Response](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/OIR_all_fuels.png?raw=true)

The impulse response analysis for total generation shows how shocks to the system propagate over time. The response patterns indicate the dynamic relationships between different components of the electricity generation system.

![Coal Impulse Response](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/OIR_Coal.png?raw=true)

Coal generation impulse responses show the declining influence of coal in the overall energy system. The responses indicate how shocks to coal generation affect other energy sources, reflecting the substitution dynamics in the energy market.

![Natural Gas Impulse Response](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/OIR_Natural_Gas.png?raw=true)

Natural gas shows strong impulse responses, reflecting its increasing role as a flexible generation source. The analysis reveals how natural gas generation responds to and influences other energy sources, demonstrating its importance in the modern energy mix.

![Nuclear Impulse Response](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/OIR_Nuclear.png?raw=true)

Nuclear power impulse responses show the stable, baseload nature of nuclear generation. The responses indicate limited interaction with other sources, reflecting nuclear power's role as a consistent generation source with minimal short-term variability.

![Other Renewables Impulse Response](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/Orthogan_impulse_response_other_renewables.png?raw=true)

Renewable energy impulse responses show the growing but still variable nature of renewable generation. The analysis reveals how renewable sources interact with conventional generation, highlighting the need for flexible backup sources.

## Machine Learning Approaches

### Random Forest Feature Engineering

Machine learning models incorporate lagged variables and engineered features to improve forecasting performance.

![Random Forest Feature Importance](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/Feature_engineering_rf.png?raw=true)

The random forest model identifies key features that drive electricity generation patterns. Lagged values of generation, seasonal indicators, and trend components emerge as important predictors. The feature importance analysis guides the selection of variables for other machine learning models.

### XGBoost Analysis

Gradient boosting methods provide another perspective on feature importance and model performance.

![XGBoost Feature Engineering](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/xgboost_feature_engineering.png?raw=true)

XGBoost feature importance analysis reveals similar patterns to the random forest model, with lagged variables and seasonal components showing high importance. The model captures non-linear relationships and interactions between features that traditional time series models might miss.

## Prophet Model Analysis

### Facebook Prophet Implementation

Prophet provides an intuitive approach to time series forecasting with built-in handling of seasonality and trend changes.

![Prophet Forecast](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/Prophet_forecast1.png?raw=true)

The Prophet model effectively captures the seasonal patterns and trend in electricity generation. The forecast includes confidence intervals and shows the model's ability to handle the complex seasonal patterns in the data.

![Prophet Components](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/Prophet_forecast2.png?raw=true)

Prophet decomposition reveals the individual components driving the forecast: trend, yearly seasonality, and residuals. This decomposition helps understand the relative contribution of different factors to electricity generation patterns.

## Model Performance and Comparison

### Comprehensive Model Evaluation

Multiple models were evaluated using standard time series forecasting metrics to identify the best performing approach.

![Model Performance Overview](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/Model_overview.png?raw=true)

The model comparison reveals that traditional time series methods, particularly ARIMA, outperform machine learning approaches for this dataset. This result highlights the importance of understanding the specific characteristics of time series data when selecting modeling approaches.

### Forecast Comparison

Different models produce varying forecast patterns, reflecting their different approaches to capturing underlying patterns.

![Forecast Comparison](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/forecast_comparison.png?raw=true)

The forecast comparison shows how different models handle the seasonal patterns and trend extrapolation. Traditional time series models tend to produce smoother forecasts, while machine learning models can capture more complex patterns but may be less stable for long-term forecasting.

## Model Performance Results

The comprehensive evaluation of six different modeling approaches reveals clear performance differences:

| Model | MAE | RMSE | MAPE (%) |
|-------|-----|------|----------|
| ARIMA | 8,525 | 11,001 | 2.34 |
| Random Forest | 9,633 | 12,134 | 2.63 |
| Prophet | 11,619 | 14,176 | 3.23 |
| XGBoost | 16,780 | 18,914 | 4.70 |
| SVR | 13,097 | 20,239 | 3.47 |
| ETS | 24,855 | 27,652 | 6.94 |

The ARIMA model demonstrates superior performance across all metrics, achieving the lowest mean absolute error (MAE) of 8,525 MWh and the best mean absolute percentage error (MAPE) of 2.34%. This result underscores the effectiveness of traditional time series methods for electricity generation forecasting.

## Future Predictions

### 12-Month Forecast

The ensemble forecasting approach combines the best-performing models to generate reliable predictions for the next 12 months.

![Forecasting Function Results](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/Forecasting_function_chart.png?raw=true)

The production-ready forecasting function generates monthly predictions with confidence intervals. The forecast shows expected seasonal patterns with peak generation during summer and winter months, consistent with historical patterns.

### Forecast Results

| Date | Forecast (MWh) | Model |
|------|----------------|-------|
| 2025-05-01 | 347,901 | Ensemble |
| 2025-06-01 | 386,597 | Ensemble |
| 2025-07-01 | 428,515 | Ensemble |
| 2025-08-01 | 423,421 | Ensemble |
| 2025-09-01 | 366,139 | Ensemble |
| 2025-10-01 | 338,500 | Ensemble |
| 2025-11-01 | 330,658 | Ensemble |
| 2025-12-01 | 365,865 | Ensemble |
| 2026-01-01 | 380,387 | Ensemble |
| 2026-02-01 | 336,649 | Ensemble |
| 2026-03-01 | 341,046 | Ensemble |
| 2026-04-01 | 318,215 | Ensemble |

## Key Findings and Insights

### Statistical Properties
The analysis reveals several important statistical characteristics of U.S. electricity generation:

- Strong seasonal patterns with 12-month cycles corresponding to weather-driven demand
- Positive long-term trend reflecting economic growth and electrification
- Stationarity achieved through first-order differencing
- Multiple structural breaks corresponding to major economic and policy changes

### Energy Mix Evolution
The 24-year analysis period captures significant changes in the U.S. energy landscape:

- Coal generation declining from dominant source to reduced role
- Natural gas becoming the primary generation source
- Nuclear maintaining stable baseload contribution
- Renewable energy showing consistent growth trajectory

### Forecasting Insights
The model comparison provides valuable insights for practitioners:

- Traditional time series methods excel for electricity generation forecasting
- ARIMA models effectively capture seasonal and trend patterns
- Machine learning models show promise but require careful tuning
- Ensemble approaches provide robust predictions with reduced model risk

## Technical Implementation

### Software and Libraries
The analysis utilizes R programming language with specialized packages for time series analysis:

- Time Series Analysis: forecast, TSstudio, tseries, prophet
- Statistical Testing: urca, vars, MTS, changepoint
- Machine Learning: randomForest, xgboost, e1071, caret
- Visualization: ggplot2, plotly, corrplot, gridExtra
- Data Manipulation: tidyverse, lubridate, dplyr

### Analytical Approach
The methodology follows established time series analysis practices:

1. Data preprocessing and cleaning
2. Exploratory analysis and visualization
3. Stationarity testing and transformation
4. Model development and parameter estimation
5. Cross-validation and performance evaluation
6. Ensemble forecasting and prediction intervals

## Business Applications

### Energy Sector Planning
The analysis provides actionable insights for energy sector stakeholders:

- Seasonal demand patterns inform capacity planning decisions
- Fuel mix trends guide investment strategies
- Volatility analysis supports risk management
- Forecasting capabilities enable operational planning

### Policy Implications
The results have important implications for energy policy:

- Renewable energy integration requires grid flexibility
- Natural gas infrastructure needs continued development
- Coal plant retirements require replacement capacity planning
- Energy storage becomes increasingly important

## Repository Structure

```
time-series-analysis/
├── data/
│   └── Net_generation_for_all_sectors.csv
├── results/
│   ├── model_performance_comparison.csv
│   ├── electricity_generation_forecast.csv
│   └── yearly_generation_trends.csv
├── visualizations/
│   └── [All analysis charts and plots]
├── .gitignore
├── LICENSE
├── README.md
├── time-series-analysis.Rproj
└── timeseries_analysis.R
```

## Conclusion

This comprehensive analysis of U.S. electricity generation data demonstrates the power of combining traditional time series methods with modern analytical techniques. The results provide valuable insights into America's evolving energy landscape and offer reliable forecasting capabilities for energy sector planning.

The superior performance of ARIMA models highlights the importance of understanding data characteristics when selecting analytical approaches. The analysis reveals significant trends in the energy mix, with clear implications for future energy policy and investment decisions.

The forecasting framework developed in this project provides a robust foundation for ongoing energy analysis and planning, with the flexibility to incorporate new data and adapt to changing market conditions.