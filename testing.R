library(fpp2)
library(forecast)
library(readxl)

# Load BTC data
btc_data <- read_excel("C:/Users/Usera/Desktop/sem6/STA572/BTC_Data_final.xlsx")

# Check column names
head(btc_data)

# Convert to time series (daily data)
btcts <- ts(btc_data$priceUSD, frequency = 365)

# 80/20 split
n <- length(btcts)
split_point <- round(n * 0.8)

est_part <- head(btcts, split_point)
eva_part <- tail(btcts, n - split_point)

# Holt's estimation (finds optimal alpha & beta)
Holt_est <- holt(est_part)
summary(Holt_est)

# Extract alpha & beta from estimation
alpha_opt <- Holt_est$model$par["alpha"]
beta_opt  <- Holt_est$model$par["beta"]

cat("Optimal Alpha:", alpha_opt, "\n")
cat("Optimal Beta:", beta_opt, "\n")

# Holt's evaluation using optimal alpha & beta
Holt_eva <- holt(eva_part, alpha = alpha_opt, beta = beta_opt)
summary(Holt_eva)

# Plot full BTC data with forecast
autoplot(btcts) + ylab("BTC Price (USD)") + xlab("Time") +
  ggtitle("BTC Price Forecast using Holt's Method") +
  autolayer(forecast(holt(btcts, alpha = alpha_opt, beta = beta_opt), h = 5))