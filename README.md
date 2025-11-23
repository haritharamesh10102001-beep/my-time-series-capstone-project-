# my-time-series-capstone-project-
Advanced LSTM time-series forecasting project with attention mechanism
Objective: Predict S&P 500 (ticker ^GSPC) daily closing prices.
Data Source: Yahoo Finance (yfinance) for S&P 500 historical data (2010-2023).
Features Used: 'Open', 'High', 'Low', 'Close', 'Volume'.
Data Preprocessing:
MinMaxScaler for feature scaling (0-1 range).
look_back window of 60 days to create input sequences.
Train (70%), Validation (15%), Test (15%) split.
Models Developed:
Baseline LSTM: Standard two-layer LSTM with Dropout and Dense output.
Attention-LSTM: Custom AttentionLayer integrated with LSTM to dynamically weigh past time steps' importance.
Training Details:
Optimizer: Adam
Loss Function: Mean Squared Error (mse)
Regularization: Early Stopping based on validation loss to prevent overfitting.
Evaluation Metrics:
Root Mean Squared Error (RMSE)
Mean Absolute Error (MAE)
Directional Accuracy (although there was a runtime warning during its calculation, the underlying logic was present).
Visualization: Plots for training history (loss), actual vs. predicted prices, and attention weights for interpretability.
Outcome: The project provides a comparative analysis of the forecasting performance between a standard LSTM and an attention-augmented LSTM on real-world financial data, including insights into which historical periods the attention mechanism focuses on.
