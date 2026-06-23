# =============================================================================
# STA572/570 - Assessment 3
# Section 5b: ARIMA/SARIMA Modelling via Box-Jenkins Methodology
# =============================================================================

library(fpp2)
library(forecast)
library(readxl)
library(tseries)    # for ADF test

# =============================================================================
# 1. LOAD DATA
# =============================================================================

btc_data <- read_excel("C:/Users/Usera/Desktop/sem6/STA572/assignment 3/BTC_Data_final.xlsx")

cat("Total observations:", nrow(btc_data), "\n")
cat("Date range:", as.character(min(btc_data$Date)), "to", as.character(max(btc_data$Date)), "\n")

btcts <- ts(btc_data$priceUSD, frequency = 365)


# =============================================================================
# 2. TRAIN / TEST SPLIT  (70% fitted, 30% hold-out)
# =============================================================================

n        <- length(btcts)
split_pt <- round(n * 0.70)
train    <- head(btcts, split_pt)
test     <- tail(btcts, n - split_pt)

cat("\nTrain size:", length(train), "| Hold-out size:", length(test), "\n")


# =============================================================================
# PART A : MODEL IDENTIFICATION
# =============================================================================

# -----------------------------------------------------------------------------
# A1. TIME SERIES PLOT (on fitted/train portion)
# -----------------------------------------------------------------------------

autoplot(train) +
  ggtitle("BTC Price - Training Set (70%)") +
  ylab("Price (USD)") + xlab("Year") +
  theme_minimal()


# -----------------------------------------------------------------------------
# A2. ACF AND PACF ON ORIGINAL TRAINING DATA
# -----------------------------------------------------------------------------
# ACF : measures correlation between the series and its own lags.
#       Slowly decaying ACF = non-stationary series.
# PACF: measures direct correlation at each lag after removing shorter lag effects.
#       Used to identify AR order.
# -----------------------------------------------------------------------------

par(mfrow = c(1, 2))
acf(train,  lag.max = 40, main = "ACF - Original BTC Price (Train)")
pacf(train, lag.max = 40, main = "PACF - Original BTC Price (Train)")
par(mfrow = c(1, 1))


# -----------------------------------------------------------------------------
# A3. STATIONARITY TESTS ON ORIGINAL SERIES
# -----------------------------------------------------------------------------
# Three tests are used together for a robust conclusion:
#
# ADF  (Augmented Dickey-Fuller):
#   H0 : Series has a unit root (non-stationary)
#   H1 : Series is stationary
#   p > 0.05 -> NON-STATIONARY | p < 0.05 -> STATIONARY
#
# PP   (Phillips-Perron):
#   H0 : Series has a unit root (non-stationary)
#   H1 : Series is stationary
#   p > 0.05 -> NON-STATIONARY | p < 0.05 -> STATIONARY
#
# KPSS (Kwiatkowski-Phillips-Schmidt-Shin):
#   H0 : Series is STATIONARY
#   H1 : Series has a unit root (non-stationary)
#   p < 0.05 -> NON-STATIONARY | p > 0.05 -> STATIONARY
#   NOTE: KPSS has OPPOSITE null hypothesis to ADF and PP
# -----------------------------------------------------------------------------

cat("\n============================================================\n")
cat("STATIONARITY TESTS - ORIGINAL SERIES\n")
cat("============================================================\n")

adf_original  <- adf.test(train)
pp_original   <- pp.test(train)
kpss_original <- kpss.test(train)

cat("\nADF Test  p-value :", round(adf_original$p.value,  4),
    ifelse(adf_original$p.value  > 0.05, "-> NON-STATIONARY", "-> STATIONARY"), "\n")
cat("PP Test   p-value :", round(pp_original$p.value,   4),
    ifelse(pp_original$p.value   > 0.05, "-> NON-STATIONARY", "-> STATIONARY"), "\n")
cat("KPSS Test p-value :", round(kpss_original$p.value, 4),
    ifelse(kpss_original$p.value < 0.05, "-> NON-STATIONARY", "-> STATIONARY"), "\n")

if (adf_original$p.value > 0.05 | pp_original$p.value > 0.05 | kpss_original$p.value < 0.05) {
  cat("\nOverall Conclusion: Series is NON-STATIONARY. Differencing is required.\n")
} else {
  cat("\nOverall Conclusion: Series is STATIONARY. No differencing needed.\n")
}


# -----------------------------------------------------------------------------
# A4. FIRST DIFFERENCING
# -----------------------------------------------------------------------------
# Differencing removes trend by computing: y't = yt - yt-1
# d=1 means first-order differencing (one round of subtraction)
# -----------------------------------------------------------------------------

train_diff1 <- diff(train, differences = 1)

autoplot(train_diff1) +
  ggtitle("BTC Price After 1st Differencing") +
  ylab("Differenced Price") + xlab("Year") +
  theme_minimal()

par(mfrow = c(1, 2))
acf(train_diff1,  lag.max = 40, main = "ACF - After 1st Difference")
pacf(train_diff1, lag.max = 40, main = "PACF - After 1st Difference")
par(mfrow = c(1, 1))

cat("\n============================================================\n")
cat("STATIONARITY TESTS - AFTER 1ST DIFFERENCING\n")
cat("============================================================\n")

adf_diff1  <- adf.test(train_diff1)
pp_diff1   <- pp.test(train_diff1)
kpss_diff1 <- kpss.test(train_diff1)

cat("\nADF Test  p-value :", round(adf_diff1$p.value,  4),
    ifelse(adf_diff1$p.value  > 0.05, "-> NON-STATIONARY", "-> STATIONARY"), "\n")
cat("PP Test   p-value :", round(pp_diff1$p.value,   4),
    ifelse(pp_diff1$p.value   > 0.05, "-> NON-STATIONARY", "-> STATIONARY"), "\n")
cat("KPSS Test p-value :", round(kpss_diff1$p.value, 4),
    ifelse(kpss_diff1$p.value < 0.05, "-> NON-STATIONARY", "-> STATIONARY"), "\n")

if (adf_diff1$p.value > 0.05 | pp_diff1$p.value > 0.05 | kpss_diff1$p.value < 0.05) {
  cat("Conclusion: Still NON-STATIONARY. Second differencing may be needed.\n")

  # Second differencing if needed
  train_diff2 <- diff(train, differences = 2)

  par(mfrow = c(1, 2))
  acf(train_diff2,  lag.max = 40, main = "ACF - After 2nd Difference")
  pacf(train_diff2, lag.max = 40, main = "PACF - After 2nd Difference")
  par(mfrow = c(1, 1))

  cat("\n============================================================\n")
  cat("ADF TEST - AFTER 2ND DIFFERENCING\n")
  cat("============================================================\n")
  adf_diff2 <- adf.test(train_diff2)
  print(adf_diff2)

  stationary_series <- train_diff2
  d_order <- 2
} else {
  cat("Conclusion: Series is STATIONARY after 1st differencing.\n")
  stationary_series <- train_diff1
  d_order <- 1
}

cat("\nDifferencing order (d) to use in ARIMA:", d_order, "\n")


# -----------------------------------------------------------------------------
# A5. IDENTIFY p AND q FROM ACF / PACF OF STATIONARY SERIES
# -----------------------------------------------------------------------------
# After differencing:
#   ACF cuts off at lag q  -> suggests MA(q) component
#   PACF cuts off at lag p -> suggests AR(p) component
#
# Based on ACF/PACF of differenced BTC data, 5 candidate models are proposed:
#   Model 1 : ARIMA(1,1,1)  - one AR, one MA term
#   Model 2 : ARIMA(2,1,2)  - two AR, two MA terms
#   Model 3 : ARIMA(1,1,0)  - pure AR(1) after differencing
#   Model 4 : ARIMA(0,1,1)  - pure MA(1) after differencing
#   Model 5 : ARIMA(2,1,1)  - two AR, one MA term
# -----------------------------------------------------------------------------

cat("\n============================================================\n")
cat("CANDIDATE ARIMA MODELS IDENTIFIED:\n")
cat("  Model 1 : ARIMA(1,1,1)\n")
cat("  Model 2 : ARIMA(2,1,2)\n")
cat("  Model 3 : ARIMA(1,1,0)\n")
cat("  Model 4 : ARIMA(0,1,1)\n")
cat("  Model 5 : ARIMA(2,1,1)\n")
cat("============================================================\n")


# =============================================================================
# PART B : MODEL ESTIMATION AND VALIDATION
# =============================================================================

# -----------------------------------------------------------------------------
# B1. FIT ALL 5 CANDIDATE ARIMA MODELS ON TRAINING SET
# -----------------------------------------------------------------------------

cat("\n============================================================\n")
cat("FITTING 5 ARIMA MODELS ON TRAINING SET\n")
cat("============================================================\n")

m1 <- Arima(train, order = c(1, 1, 1))
m2 <- Arima(train, order = c(2, 1, 2))
m3 <- Arima(train, order = c(1, 1, 0))
m4 <- Arima(train, order = c(0, 1, 1))
m5 <- Arima(train, order = c(2, 1, 1))

cat("\nModel 1 ARIMA(1,1,1) - AIC:", round(m1$aic, 2), "| BIC:", round(m1$bic, 2), "\n")
cat("Model 2 ARIMA(2,1,2) - AIC:", round(m2$aic, 2), "| BIC:", round(m2$bic, 2), "\n")
cat("Model 3 ARIMA(1,1,0) - AIC:", round(m3$aic, 2), "| BIC:", round(m3$bic, 2), "\n")
cat("Model 4 ARIMA(0,1,1) - AIC:", round(m4$aic, 2), "| BIC:", round(m4$bic, 2), "\n")
cat("Model 5 ARIMA(2,1,1) - AIC:", round(m5$aic, 2), "| BIC:", round(m5$bic, 2), "\n")


# -----------------------------------------------------------------------------
# B2. MODEL SELECTION TABLE (AIC, BIC, LJUNG-BOX)
# -----------------------------------------------------------------------------
# AIC  : Akaike Information Criterion  - lower is better (balances fit vs complexity)
# BIC  : Bayesian Information Criterion - lower is better (penalises complexity more)
# Ljung-Box test:
#   H0 : Residuals are white noise (no autocorrelation) -> GOOD model
#   H1 : Residuals are autocorrelated -> model is inadequate
#   p-value > 0.05 -> residuals are white noise -> model passed diagnostic
# -----------------------------------------------------------------------------

models      <- list(m1, m2, m3, m4, m5)
model_names <- c("ARIMA(1,1,1)", "ARIMA(2,1,2)", "ARIMA(1,1,0)", "ARIMA(0,1,1)", "ARIMA(2,1,1)")

aic_vals <- sapply(models, function(m) round(m$aic, 2))
bic_vals <- sapply(models, function(m) round(m$bic, 2))
lb_pvals <- sapply(models, function(m) {
  lb <- Box.test(residuals(m), lag = 20, type = "Ljung-Box")
  round(lb$p.value, 4)
})
lb_result <- ifelse(lb_pvals > 0.05, "PASS (white noise)", "FAIL (autocorrelated)")

selection_table <- data.frame(
  Model          = model_names,
  AIC            = aic_vals,
  BIC            = bic_vals,
  LjungBox_p     = lb_pvals,
  Residual_Check = lb_result
)

cat("\n============================================================\n")
cat("MODEL SELECTION TABLE\n")
cat("============================================================\n")
print(selection_table)

# Identify best model by lowest AIC
best_idx   <- which.min(aic_vals)
best_model <- models[[best_idx]]
best_name  <- model_names[best_idx]

cat("\nBest ARIMA model (lowest AIC):", best_name, "\n")
cat("AIC:", round(best_model$aic, 2), "| BIC:", round(best_model$bic, 2), "\n")
summary(best_model)


# -----------------------------------------------------------------------------
# B3. RESIDUAL DIAGNOSTICS ON BEST MODEL
# -----------------------------------------------------------------------------

cat("\n============================================================\n")
cat("RESIDUAL DIAGNOSTICS -", best_name, "\n")
cat("============================================================\n")

checkresiduals(best_model)

lb_best <- Box.test(residuals(best_model), lag = 20, type = "Ljung-Box")
cat("\nLjung-Box Test p-value:", round(lb_best$p.value, 4), "\n")
if (lb_best$p.value > 0.05) {
  cat("Conclusion: Residuals are WHITE NOISE. Model is adequate.\n")
} else {
  cat("Conclusion: Residuals show autocorrelation. Model may need revision.\n")
}


# -----------------------------------------------------------------------------
# B4. FORECAST ON HOLD-OUT SET USING BEST MODEL
# -----------------------------------------------------------------------------

arima_forecast <- forecast(best_model, h = length(test))

autoplot(btcts) +
  autolayer(arima_forecast$mean, series = paste(best_name, "Forecast"), PI = FALSE) +
  autolayer(fitted(best_model),  series = paste(best_name, "Fitted"),   PI = FALSE) +
  ggtitle(paste("Best ARIMA Model:", best_name)) +
  ylab("BTC Price (USD)") + xlab("Year") +
  theme_minimal()


# -----------------------------------------------------------------------------
# B5. ERROR MEASURES ON HOLD-OUT
# -----------------------------------------------------------------------------

calc_MSE <- function(actual, fitted) {
  mean((actual - fitted)^2, na.rm = TRUE)
}

calc_MAPE <- function(actual, fitted) {
  mean(abs((actual - fitted) / actual) * 100, na.rm = TRUE)
}

arima_test_MSE  <- calc_MSE(as.numeric(test),  as.numeric(arima_forecast$mean))
arima_test_MAPE <- calc_MAPE(as.numeric(test), as.numeric(arima_forecast$mean))

cat("\n============================================================\n")
cat("HOLD-OUT ERROR MEASURES -", best_name, "\n")
cat("============================================================\n")
cat("MSE :", round(arima_test_MSE,  2), "\n")
cat("MAPE:", round(arima_test_MAPE, 4), "%\n")


# -----------------------------------------------------------------------------
# B6. ONE-STEP-AHEAD FORECAST
# -----------------------------------------------------------------------------

best_model_full <- Arima(btcts, order = arimaorder(best_model))
arima_onestep   <- forecast(best_model_full, h = 1)

cat("\n============================================================\n")
cat("ONE-STEP-AHEAD FORECAST -", best_name, "\n")
cat("============================================================\n")
cat("Forecast:", round(arima_onestep$mean, 2), "USD\n")
cat("80% CI  : [", round(arima_onestep$lower[1], 2), ",", round(arima_onestep$upper[1], 2), "]\n")
cat("95% CI  : [", round(arima_onestep$lower[2], 2), ",", round(arima_onestep$upper[2], 2), "]\n")


# =============================================================================
# PART C : COMPARE BEST ARIMA vs BEST UNIVARIATE (ARRES)
# =============================================================================
# From Section 5a, ARRES achieved:
#   Hold-out MSE  = 2,647,173
#   Hold-out MAPE = 3.08%
# =============================================================================

arres_holdout_MSE  <- 2647173
arres_holdout_MAPE <- 3.0820

cat("\n============================================================\n")
cat("FINAL COMPARISON: BEST ARIMA vs BEST UNIVARIATE (ARRES)\n")
cat("============================================================\n")

final_comparison <- data.frame(
  Model        = c(best_name, "ARRES"),
  Holdout_MSE  = round(c(arima_test_MSE,  arres_holdout_MSE),  2),
  Holdout_MAPE = round(c(arima_test_MAPE, arres_holdout_MAPE), 4)
)

print(final_comparison)

if (arima_test_MAPE < arres_holdout_MAPE) {
  cat("\nConclusion:", best_name, "outperforms ARRES. ARIMA is the recommended model.\n")
} else {
  cat("\nConclusion: ARRES outperforms", best_name, ". ARRES remains the recommended model.\n")
}
