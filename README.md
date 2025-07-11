# Comprehensive Time Series Analysis of Electricity Generation Data

## Project Overview

This project analyzes U.S. electricity generation patterns using 24 years of monthly data from January 2001 to April 2025. The analysis combines traditional time series methods with modern machine learning techniques to understand energy trends and forecast future electricity generation across different fuel sources.

The dataset contains 292 monthly observations covering 11 different energy sources including coal, natural gas, nuclear, hydroelectric, and renewable energy. This comprehensive analysis reveals significant insights about America's evolving energy landscape and provides accurate forecasting capabilities for energy sector planning.

## Data Analysis and Methodology

### Initial Data Exploration

The analysis begins with exploratory data analysis to understand the underlying patterns in electricity generation. The time series exhibits strong seasonal patterns with peak demand typically occurring during summer and winter months due to increased cooling and heating needs.

![Trend Analysis](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/Trend.png?raw=true)

Long-term trend analysis reveals consistent growth in total electricity generation over the 24-year period, with notable shifts in the energy mix composition. The data shows a clear transition from coal-dependent generation toward natural gas and renewable sources, reflecting both market dynamics and policy changes in the energy sector.

### Seasonal and Temporal Patterns

Understanding seasonal variations is crucial for energy planning and grid management. The analysis reveals distinct monthly and quarterly patterns that align with known demand cycles in the electricity sector.

![Monthly and Yearly Distribution](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/monthly_yearly_distribution.png?raw=true)

The seasonal analysis shows predictable patterns with higher generation during summer months (June-August) and winter months (December-February), corresponding to peak air conditioning and heating demands. These patterns are consistent across years but show some variation in magnitude, reflecting economic growth and population changes.

### Time Series Decomposition

Decomposition analysis separates the time series into trend, seasonal and residual components, providing insight into the underlying structure of electricity generation patterns.

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


The correlation analysis reveals complex relationships between different electricity generation sources. The heatmap indicates that Pumped Storage has a strong negative correlation (around -0.8) with Conventional Hydro, suggesting an opposing role in energy management. Natural Gas exhibits a strong positive correlation (around 0.8) with Coal, reflecting a close relationship possibly tied to market dynamics. Coal and Other Renewables show mixed correlations, with Coal aligning positively with Natural Gas and Other Renewables having moderate positive ties (e.g., ~0.6 with Petroleum Liquids), while substitution effects appear limited. Nuclear power displays moderate correlations (around 0.2 to 0.4), consistent with its stable baseload generation role.


### Autocorrelation Analysis

Autocorrelation analysis helps identify the appropriate model structure for time series forecasting.

![ACF and PACF Analysis](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/ACF_PACF.png?raw=true)

The provided plots display Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) for electricity generation data across all fuels, both in their original form and after differencing.

ACF - All Fuels: The ACF plot shows autocorrelation coefficients close to zero for all lags (up to 3), with values fluctuating slightly   around the confidence interval (blue dashed lines). This suggests little to no significant autocorrelation in the original data, indicating it may not be stationary.
PACF - All Fuels: The PACF plot also shows coefficients near zero for most lags, except for a potential significant spike at lag 1. This could imply a slight direct relationship with the immediate past value, but overall autocorrelation appears weak.
ACF - Differenced All Fuels: After differencing, the ACF plot shows coefficients remaining close to zero across all lags, with no significant spikes beyond the confidence interval. This indicates that differencing has helped remove any trend or autocorrelation, suggesting the data is now stationary.
PACF - Differenced All Fuels: The PACF plot for differenced data shows a significant spike at lag 1, with subsequent lags close to zero. This suggests that the differenced series may follow an AR(1) process, where only the first lag has a notable influence.

### Change Point Detection

Identifying structural changes in the time series helps understand major shifts in electricity generation patterns.

![Change Point Analysis](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/Change_point_detction.png?raw=true)

The change point detection plot for all fuels generation from 2000 to 2025 shows the total electricity generation (in MW) fluctuating around a mean level of approximately 35,000 MW, indicated by the red horizontal line. The data exhibits significant variability, with peaks reaching above 40,000 MW and dips below 30,000 MW, suggesting seasonal or operational influences. No clear structural breaks or abrupt shifts in the mean level are evident across the observed period, implying a relatively stable overall trend despite the fluctuations. This stability, combined with the consistent mean, indicates that while short-term variations occur, there are no major long-term changes in the generation pattern up to July 2025.

### Regime Switching Analysis

Regime switching models capture different market conditions and operational states in electricity generation.

![Regime 1 Probabilities](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/Regime_1.png?raw=true)


The plot displays smoothed and filtered probabilities for two regimes of electricity generation data over approximately 300 time periods. For Regime 1, the smoothed probabilities (gray) and filtered probabilities (red) fluctuate around 0.5 to 0.6, with occasional peaks and dips, indicating a moderate likelihood of this regime being active throughout the series. Regime 2 shows smoothed and filtered probabilities generally ranging from 0.4 to 0.5, with similar variability, suggesting it is the complementary state to Regime 1. The alternating dominance of these probabilities over time implies a dynamic switching between the two regimes, reflecting changes in generation patterns or underlying data characteristics, with no single regime consistently dominating the entire period.


![Regime 2 Probabilities](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/Regime_2.png?raw=true)

The plot displays smoothed and filtered probabilities for two regimes of electricity generation data over approximately 300 time periods. For Regime 1, the smoothed probabilities (gray) and filtered probabilities (red) fluctuate around 0.5 to 0.6, with occasional peaks and dips, indicating a moderate likelihood of this regime being active throughout the series. Regime 2 shows smoothed and filtered probabilities generally ranging from 0.4 to 0.5, with similar variability, suggesting it is the complementary state to Regime 1. The alternating dominance of these probabilities over time implies a dynamic switching between the two regimes, reflecting changes in generation patterns or underlying data characteristics, with no single regime consistently dominating the entire period.

### Spectral Analysis

Frequency domain analysis provides additional insights into cyclical patterns in electricity generation.

![Periodogram](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/periodogram.png?raw=true)

The periodogram for all fuels generation displays the spectral density (spectrum) of the time series across a frequency range from 0 to 6, with a bandwidth of 0.0115. The plot shows several peaks in spectral power, with the most prominent occurring around frequencies 1 to 2, where the spectrum reaches values exceeding 100,000,000. This indicates strong periodic components in the data, likely corresponding to seasonal or cyclical patterns in electricity generation, such as annual or semi-annual cycles. A notable peak near frequency 5.5, marked by a blue vertical line, suggests an additional high-frequency component, possibly reflecting shorter-term fluctuations (e.g., weekly or daily variations). The overall variability across the frequency spectrum highlights the presence of multiple periodic influences on the generation data.

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

The provided plot effectively illustrates the variable importance in a Random Forest model for predicting electricity generation, using both %IncMSE and IncNodePurity metrics. Both metrics consistently highlight seasonal_lag12 and lag12 as the most crucial predictors, demonstrating their significant influence on electricity generation due to strong annual seasonality. While other lagged variables (e.g., lag3, lag5) show moderate importance, temporal indicators such as month, quarter, and trend consistently exhibit the least impact on the model's predictive accuracy. This analysis underscores the dominant role of seasonal and lagged patterns in forecasting electricity generation, with less direct influence from general temporal markers.

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

### Model Evaluation

Multiple models were evaluated using standard time series forecasting metrics to identify the best performing approach.

![Model Performance Overview](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/Model_overview.png?raw=true)

The image presents a statistical analysis of electricity generation data through four plots. The "Electricity Generation Time Series" plot (top-left) displays a time series from 2005 to 2025, with generation values (MW/h) exhibiting volatility around a stable mean, approximated by a red trend line, suggesting no significant long-term trend but notable seasonal or random fluctuations. The "Seasonal Component" plot (top-right) quantifies the seasonal effect, showing a zero-centered periodic pattern with an amplitude of approximately ±4000 MW/h, indicating consistent seasonal variability. The "Forecast vs Actual" plot (bottom-left) compares actual generation (black) with ARIMA (blue) and ETS (red) forecasts from 2021 to 2025, revealing forecast errors as deviations between lines, with ARIMA and ETS showing differing predictive precision, potentially measurable by mean absolute error or root mean square error (RMSE). The "Model Performance (RMSE)" plot (bottom-right) provides RMSE values for six models—ARIMA, Random Forest, Prophet, XGBoost, SVR, and ETS—ranging from approximately 1000 to 2500, with ETS exhibiting the highest RMSE (indicating poorer fit) and ARIMA the lowest (suggesting better accuracy), allowing for a statistical comparison of model efficacy based on forecast error metrics. This analysis suggests that ARIMA may be the most reliable model for forecasting electricity generation, while ETS underperforms, and the consistent seasonal component implies that seasonal adjustments are crucial for accurate predictions, potentially guiding future modeling efforts or energy planning strategies.

### Forecast Comparison

Different models produce varying forecast patterns, reflecting their different approaches to capturing underlying patterns.

![Forecast Comparison](https://github.com/farukhasan/time-series-analysis/blob/main/visualizations/forecast_comparison.png?raw=true)

The plot presents a statistical analysis of electricity generation (MW/h) from 2021 to 2025, comparing actual values (solid red line) with forecasts from ARIMA (green dashed), ETS (cyan dashed), and Prophet (purple dashed) models. The actual data exhibits a cyclical pattern with an amplitude of approximately 5000 MW/h, suggesting strong seasonal or periodic influences. The forecasts deviate from the actual values, with varying degrees of accuracy—ARIMA and ETS appear to capture the general trend but with noticeable errors, while Prophet shows larger discrepancies, particularly in peak and trough predictions, indicating potential model misspecification. Statistically, the RMSE or mean absolute error could quantify these deviations, with the plot suggesting that ARIMA and ETS may outperform Prophet in tracking the observed variability. This implies that ARIMA and ETS might be more suitable for short-term forecasting, while the consistent cyclical nature of the data underscores the importance of incorporating seasonal components in future models.

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