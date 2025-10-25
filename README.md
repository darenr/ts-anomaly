# Time Series Anomaly Detection with LSTM

A real-time anomaly detection system for time series data using LSTM (Long Short-Term Memory) neural networks. This project demonstrates how to detect unusual patterns in streaming data with seasonal components.

## Overview

This project implements an LSTM-based anomaly detector that can identify outliers in time series data with both daily and weekly seasonal patterns. It's designed to work with streaming data and provides real-time visualization of detected anomalies.

## Features

- **LSTM Neural Network**: Deep learning model that learns temporal patterns and dependencies
- **Multi-Feature Input**: Incorporates time-based features (day of week, hour) alongside raw values
- **Adaptive Threshold**: Uses percentile-based error thresholding for flexible anomaly detection
- **Real-Time Processing**: Detects anomalies in streaming data using a sliding window approach
- **Live Visualization**: Animated plots showing actual values, predictions, and detected anomalies
- **Seasonal Pattern Support**: Handles both daily and weekly cyclical patterns in data

## How It Works

1. **Training Phase**: The model trains on historical data to learn normal patterns
2. **Detection Phase**: New data points are continuously evaluated against predictions
3. **Anomaly Identification**: Points with prediction errors exceeding the threshold are flagged as anomalies
4. **Visualization**: Real-time matplotlib animation displays the stream with anomalies highlighted in red

## Use Cases

This implementation is ideal for:

- Website performance monitoring (page load times)
- Server metrics monitoring (CPU, memory usage)
- Network traffic analysis
- IoT sensor data monitoring
- Any time series data with periodic patterns

## Technical Details

- **Lookback Period**: 168 time steps (1 week of hourly data by default)
- **Architecture**: 2-layer LSTM with 64 hidden units
- **Input Features**: Value, day of week, hour of day
- **Anomaly Threshold**: 95th percentile of training errors (configurable)

## Requirements

See `pyproject.toml` for dependencies:

- PyTorch (deep learning framework)
- NumPy (numerical computing)
- Pandas (data manipulation)
- Scikit-learn (preprocessing)
- Matplotlib (visualization)

## Running the Demo

The demo simulates page load time monitoring with synthetic data:

```bash
python main.py
```

This will:

1. Generate synthetic time series data with daily and weekly cycles
2. Train the LSTM model on historical patterns
3. Launch a live animation showing real-time anomaly detection

Anomalies are artificially introduced every 50 time steps to demonstrate the detection capability.

## License

See LICENSE file for details.
