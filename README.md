# TSAF-Ex-9-The-Box-Jenkins-Model-ARIMA


## AIM:
To understand and implement the Box-Jenkins approach to time-series forecasting using an
ARIMA model in Python. 

## PROGRAM:

```
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

subscriptions = [
 15, 18, 17, 21, 24, 23, 27, 30,
 29, 33, 36, 35, 39, 42, 41, 45,
 48, 47, 51, 54, 53, 57, 60, 59
]
data = pd.Series(subscriptions, name="Subscriptions")

print("STEP 1: Original Subscriptions Data")
print(data.to_string(index=False))
plt.figure(figsize=(10, 4))
plt.plot(data, marker="o")
plt.title("Monthly Active Subscriptions Data")
plt.xlabel("Month")
plt.ylabel("Subscriptions (in thousands)")
plt.grid(True)
plt.tight_layout()
plt.show()
adf_result = adfuller(data)

print("\nSTEP 2: ADF Test on Original Data")
print("ADF Statistic:", round(adf_result[0], 4))
print("p-value:", round(adf_result[1], 4))
if adf_result[1] < 0.05:
 print("Conclusion: Evidence of stationarity.")
else:
 print("Conclusion: Insufficient evidence of stationarity.")
diff_data = data.diff().dropna()

print("\nSTEP 3: First-Differenced Data")
print(diff_data.to_string(index=False))
plt.figure(figsize=(10, 4))
plt.plot(diff_data, marker="o")
plt.title("First-Differenced Subscriptions")
plt.xlabel("Month")
plt.ylabel("Change in Subscriptions")
plt.grid(True)
plt.tight_layout()
plt.show()
adf_diff = adfuller(diff_data)

print("\nSTEP 4: ADF Test After Differencing")
print("ADF Statistic:", round(adf_diff[0], 4))
print("p-value:", round(adf_diff[1], 4))
if adf_diff[1] < 0.05:
 print("Conclusion: Evidence of stationarity.")
else:
 print("Conclusion: Stationarity is not confirmed.")

print("\nSTEP 5: ACF and PACF Plots")
plot_acf(diff_data, lags=10)
plt.tight_layout()
plt.show()
plot_pacf(diff_data, lags=10, method="ywm")
plt.tight_layout()
plt.show()
model = ARIMA(data, order=(1, 1, 1))
fitted_model = model.fit()

print("\nSTEP 6: ARIMA Model Summary")
print(fitted_model.summary())

print("\nSTEP 7: Model Selection Criteria")
print("AIC:", round(fitted_model.aic, 4))
print("BIC:", round(fitted_model.bic, 4))
residuals = fitted_model.resid
plt.figure(figsize=(10, 4))
plt.plot(residuals, marker="o")
plt.axhline(0, color="red", linestyle="--")
plt.title("ARIMA Residuals")
plt.xlabel("Month")
plt.ylabel("Residual")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nSTEP 8: Residual ACF")
plot_acf(residuals, lags=10)
plt.tight_layout()
plt.show()
lb_result = acorr_ljungbox(residuals, lags=[5], return_df=True)
print("\nLjung-Box Test Result")
print(lb_result)
if lb_result["lb_pvalue"].iloc[0] > 0.05:
 print("No significant residual autocorrelation detected at lag 5.")
else:
 print("Residual autocorrelation may remain; review the model.")
forecast = fitted_model.get_forecast(steps=6)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

print("\nSTEP 9: Forecast for the Next Six Months")
for month, value in enumerate(forecast_mean, start=len(data) + 1):
 print(f"Month {month}: {value:.2f}")
print("\nForecast Confidence Intervals")
print(forecast_ci.round(2))
future_months = range(len(data), len(data) + 6)
plt.figure(figsize=(10, 5))
plt.plot(range(len(data)), data, marker="o", label="Actual Subscriptions")
plt.plot(future_months, forecast_mean.to_numpy(), marker="o", linestyle="--",
label="Forecast")
plt.fill_between(
 future_months,
 forecast_ci.iloc[:, 0].to_numpy(),
 forecast_ci.iloc[:, 1].to_numpy(),
 alpha=0.2,
 label="95% Confidence Interval"
)
plt.title("Box-Jenkins ARIMA: Six-Month Subscriptions Forecast")
plt.xlabel("Month")
plt.ylabel("Subscriptions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print("\nRESULT")
print("Box-Jenkins ARIMA modelling completed successfully.")

```

## OUTPUT:

<img width="460" height="576" alt="image" src="https://github.com/user-attachments/assets/f1466e5b-7fcd-4c21-9ea5-19b159a791b7" />

<img width="462" height="622" alt="image" src="https://github.com/user-attachments/assets/1bd91b0b-1d27-4cc6-b4ac-a582ec10b760" />

<img width="461" height="566" alt="image" src="https://github.com/user-attachments/assets/da0752b4-4ce4-49d3-aaf8-3ced786bef6e" />

<img width="460" height="532" alt="image" src="https://github.com/user-attachments/assets/44403f4e-9148-48c3-929f-73f765d921ef" />

<img width="490" height="617" alt="image" src="https://github.com/user-attachments/assets/9d67c0be-befb-4ccb-a391-20461c209a73" />
