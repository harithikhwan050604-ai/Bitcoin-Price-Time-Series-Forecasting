# =============================================================================
# STA572/570 - Assessment 3
# Univariate Modelling Techniques for Bitcoin (BTC) Price Forecasting
# Models: Holt's Method, Single Exponential Smoothing (SES), ARRES
# =============================================================================

library(fpp2)
library(forecast)
library(readxl)

# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================

btc_data <- read_excel("C:/Users/Usera/Desktop/sem6/STA572/assignment 3/BTC_Data_final.xlsx")

cat("Total observations:", nrow(btc_data), "\n")
cat("Date range:", as.character(min(btc_data$Date)), "to", as.character(max(btc_data$Date)), "\n")

# Convert to time series object (daily data, frequency = 365)
btcts <- ts(btc_data$priceUSD, frequency = 365)

# =============================================================================
# 2. DATA DESCRIPTION - TIME SERIES PLOT
# =============================================================================

autoplot(btcts) +
  ggtitle("Bitcoin (BTC) Daily Closing Price") +
  ylab("Price (USD)") +
  xlab("Year") +
  theme_minimal()


# =============================================================================
# 3. TRAIN / TEST SPLIT  (70% fitted, 30% hold-out)
# =============================================================================

n          <- length(btcts)
split_pt   <- round(n * 0.70)
train      <- head(btcts, split_pt)
test       <- tail(btcts, n - split_pt)

cat("\nTrain size:", length(train), "| Test (hold-out) size:", length(test), "\n")


# =============================================================================
# 4. ERROR MEASURE FUNCTIONS
# =============================================================================

calc_MSE <- function(actual, fitted) {
  mean((actual - fitted)^2, na.rm = TRUE)
}

calc_MAPE <- function(actual, fitted) {
  mean(abs((actual - fitted) / actual) * 100, na.rm = TRUE)
}


# =============================================================================
# MODEL 1 : SINGLE EXPONENTIAL SMOOTHING (SES)
# -----------------------------------------------------------------------------
# Purpose : Captures the LEVEL of the series only. Suitable for data with
#           no clear trend or seasonality. Good baseline for BTC in periods
#           of relatively flat movement.
# Initial value: l0 is set to the first observation (default in ses()).
#                The optimal alpha is found via maximum likelihood estimation.
# =============================================================================

cat("\n============================================================\n")
cat("MODEL 1 : Single Exponential Smoothing (SES)\n")
cat("============================================================\n")

ses_fit  <- ses(train, h = length(test))
alpha_ses <- ses_fit$model$par["alpha"]

cat("Optimal Alpha (alpha):", round(alpha_ses, 4), "\n")
cat("Initial Level (l0)   :", round(ses_fit$model$initstate, 4), "\n\n")

# Fitted values on training set
ses_fitted <- fitted(ses_fit)

# Forecast on hold-out
ses_forecast <- ses_fit$mean

# Error measures
ses_train_MSE  <- calc_MSE(train,  ses_fitted)
ses_train_MAPE <- calc_MAPE(train, ses_fitted)
ses_test_MSE   <- calc_MSE(test,   ses_forecast)
ses_test_MAPE  <- calc_MAPE(test,  ses_forecast)

cat("--- Training Set Errors ---\n")
cat("MSE :", round(ses_train_MSE, 4), "\n")
cat("MAPE:", round(ses_train_MAPE, 4), "%\n\n")

cat("--- Hold-out Set Errors ---\n")
cat("MSE :", round(ses_test_MSE, 4), "\n")
cat("MAPE:", round(ses_test_MAPE, 4), "%\n\n")

# One-step-ahead forecast
ses_onestep <- forecast(ses(btcts, alpha = alpha_ses), h = 1)
cat("One-step-ahead Forecast (SES):", round(ses_onestep$mean, 2), "USD\n")

# Plot
autoplot(btcts) +
  autolayer(ses_fit$mean, series = "SES Forecast", PI = FALSE) +
  autolayer(ses_fitted,   series = "SES Fitted",   PI = FALSE) +
  ggtitle("Model 1: Single Exponential Smoothing") +
  ylab("BTC Price (USD)") + xlab("Year") +
  theme_minimal()


# =============================================================================
# MODEL 2 : HOLT'S METHOD (Double Exponential Smoothing)
# -----------------------------------------------------------------------------
# Purpose : Extends SES by adding a TREND component (beta). This makes it
#           better suited for BTC which shows a strong long-term upward trend.
#           Uses two smoothing parameters: alpha (level) and beta (trend).
# Initial values: l0 and b0 estimated via optimisation (default in holt()).
#                 Optimal alpha and beta found by minimising SSE.
# =============================================================================

cat("\n============================================================\n")
cat("MODEL 2 : Holt's Method (Double Exponential Smoothing)\n")
cat("============================================================\n")

holt_fit    <- holt(train, h = length(test))
alpha_holt  <- holt_fit$model$par["alpha"]
beta_holt   <- holt_fit$model$par["beta"]

cat("Optimal Alpha (alpha):", round(alpha_holt, 4), "\n")
cat("Optimal Beta  (beta) :", round(beta_holt,  4), "\n")
cat("Initial Level (l0)   :", round(holt_fit$model$initstate[1], 4), "\n")
cat("Initial Trend (b0)   :", round(holt_fit$model$initstate[2], 4), "\n\n")

holt_fitted   <- fitted(holt_fit)
holt_forecast <- holt_fit$mean

holt_train_MSE  <- calc_MSE(train,  holt_fitted)
holt_train_MAPE <- calc_MAPE(train, holt_fitted)
holt_test_MSE   <- calc_MSE(test,   holt_forecast)
holt_test_MAPE  <- calc_MAPE(test,  holt_forecast)

cat("--- Training Set Errors ---\n")
cat("MSE :", round(holt_train_MSE, 4), "\n")
cat("MAPE:", round(holt_train_MAPE, 4), "%\n\n")

cat("--- Hold-out Set Errors ---\n")
cat("MSE :", round(holt_test_MSE, 4), "\n")
cat("MAPE:", round(holt_test_MAPE, 4), "%\n\n")

holt_onestep <- forecast(holt(btcts, alpha = alpha_holt, beta = beta_holt), h = 1)
cat("One-step-ahead Forecast (Holt's):", round(holt_onestep$mean, 2), "USD\n")

autoplot(btcts) +
  autolayer(holt_fit$mean, series = "Holt Forecast", PI = FALSE) +
  autolayer(holt_fitted,   series = "Holt Fitted",   PI = FALSE) +
  ggtitle("Model 2: Holt's Method") +
  ylab("BTC Price (USD)") + xlab("Year") +
  theme_minimal()


# =============================================================================
# MODEL 3 : ADAPTIVE RESPONSE RATE EXPONENTIAL SMOOTHING (ARRES)
# -----------------------------------------------------------------------------
# Purpose : A self-adjusting version of SES where alpha changes at each time
#           step based on recent forecast errors. When errors are large, alpha
#           increases (faster response). When errors are small, alpha decreases
#           (more smoothing). Ideal for volatile series like BTC.
# Note    : There is NO "best parameter estimate" for ARRES — the alpha adapts
#           dynamically. Only the initial alpha (alpha0) is set by the analyst.
# Initial values: alpha0 = 0.1 (common starting value), l0 = first observation.
# =============================================================================

cat("\n============================================================\n")
cat("MODEL 3 : ARRES (Adaptive Response Rate Exponential Smoothing)\n")
cat("============================================================\n")

arres <- function(y, alpha0 = 0.1) {
  n       <- length(y)
  alpha   <- numeric(n)
  fitted  <- numeric(n)
  E       <- numeric(n)    # smoothed error numerator
  A       <- numeric(n)    # smoothed absolute error denominator

  alpha[1]  <- alpha0
  fitted[1] <- y[1]        # initial level = first observation
  E[1]      <- 0
  A[1]      <- 0

  for (t in 2:n) {
    e_t      <- y[t - 1] - fitted[t - 1]   # forecast error at t-1
    E[t]     <- alpha0 * e_t + (1 - alpha0) * E[t - 1]
    A[t]     <- alpha0 * abs(e_t) + (1 - alpha0) * A[t - 1]
    alpha[t] <- ifelse(A[t] == 0, alpha0, abs(E[t] / A[t]))
    alpha[t] <- min(max(alpha[t], 0.01), 0.99)   # bound [0.01, 0.99]
    fitted[t] <- alpha[t] * y[t - 1] + (1 - alpha[t]) * fitted[t - 1]
  }

  list(fitted = fitted, alpha = alpha)
}

arres_train <- arres(as.numeric(train), alpha0 = 0.1)
arres_fitted <- arres_train$fitted

arres_train_MSE  <- calc_MSE(as.numeric(train),  arres_fitted)
arres_train_MAPE <- calc_MAPE(as.numeric(train), arres_fitted)

cat("Initial alpha0 :", 0.1, "\n")
cat("Final alpha (last observation):", round(tail(arres_train$alpha, 1), 4), "\n\n")

cat("--- Training Set Errors ---\n")
cat("MSE :", round(arres_train_MSE, 4), "\n")
cat("MAPE:", round(arres_train_MAPE, 4), "%\n\n")

# Evaluate on hold-out: extend ARRES on full series up to test
arres_full   <- arres(as.numeric(btcts), alpha0 = 0.1)
arres_test_fitted <- tail(arres_full$fitted, length(test))

arres_test_MSE  <- calc_MSE(as.numeric(test),  arres_test_fitted)
arres_test_MAPE <- calc_MAPE(as.numeric(test), arres_test_fitted)

cat("--- Hold-out Set Errors ---\n")
cat("MSE :", round(arres_test_MSE, 4), "\n")
cat("MAPE:", round(arres_test_MAPE, 4), "%\n\n")

# One-step-ahead forecast for ARRES
last_alpha  <- tail(arres_full$alpha, 1)
last_fitted <- tail(arres_full$fitted, 1)
last_actual <- tail(as.numeric(btcts), 1)
arres_onestep <- last_alpha * last_actual + (1 - last_alpha) * last_fitted

cat("One-step-ahead Forecast (ARRES):", round(arres_onestep, 2), "USD\n")

# Plot ARRES
arres_ts <- ts(arres_full$fitted, frequency = 365)
autoplot(btcts, series = "Actual") +
  autolayer(arres_ts, series = "ARRES Fitted", PI = FALSE) +
  ggtitle("Model 3: ARRES") +
  ylab("BTC Price (USD)") + xlab("Year") +
  theme_minimal()


# =============================================================================
# 5. MODEL COMPARISON SUMMARY
# =============================================================================

cat("\n============================================================\n")
cat("MODEL COMPARISON SUMMARY\n")
cat("============================================================\n")

comparison <- data.frame(
  Model        = c("SES", "Holt's Method", "ARRES"),
  Train_MSE    = round(c(ses_train_MSE,  holt_train_MSE,  arres_train_MSE),  2),
  Train_MAPE   = round(c(ses_train_MAPE, holt_train_MAPE, arres_train_MAPE), 4),
  Holdout_MSE  = round(c(ses_test_MSE,   holt_test_MSE,   arres_test_MSE),   2),
  Holdout_MAPE = round(c(ses_test_MAPE,  holt_test_MAPE,  arres_test_MAPE),  4)
)

print(comparison)

best_model <- comparison$Model[which.min(comparison$Holdout_MSE)]
cat("\nBest model based on Hold-out MSE:", best_model, "\n")

best_mape  <- comparison$Model[which.min(comparison$Holdout_MAPE)]
cat("Best model based on Hold-out MAPE:", best_mape, "\n")


# =============================================================================
# 6. ONE-STEP-AHEAD FORECAST SUMMARY
# =============================================================================

cat("\n============================================================\n")
cat("ONE-STEP-AHEAD FORECAST (next day after last observation)\n")
cat("============================================================\n")

cat("SES    :", round(ses_onestep$mean,  2), "USD\n")
cat("Holt's :", round(holt_onestep$mean, 2), "USD\n")
cat("ARRES  :", round(arres_onestep,     2), "USD\n")
