import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import deque
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class LSTMAnomalyDetector(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=1):  # Increased input_size for features
        super(LSTMAnomalyDetector, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

class TimeSeriesAnomalyDetector:
    def __init__(self, lookback_period=168, prediction_window=1, anomaly_percentile=95):
        self.lookback_period = lookback_period  # 1 week of hourly data
        self.prediction_window = prediction_window
        self.anomaly_percentile = anomaly_percentile  # Use 95th percentile of errors
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()  # For day/hour features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMAnomalyDetector(input_size=3).to(self.device)  # 3 inputs: value, day, hour
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.history = deque(maxlen=lookback_period)
        self.error_threshold = 0  # Will be set after training
        self.time_points = []
        self.values = []
        self.predictions = []
        self.anomalies = []
        
    def prepare_data(self, data, times):
        X, y = [], []
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1))
        # Extract day of week (0-6) and hour (0-23) from times
        days = np.array([t % 7 for t in times])  # Assuming times are in days
        hours = np.array([(t * 24 % 24) for t in times])
        features = np.column_stack((days, hours))
        features_scaled = self.feature_scaler.fit_transform(features)
        
        for i in range(len(data_scaled) - self.lookback_period - self.prediction_window + 1):
            sequence = np.hstack((
                data_scaled[i:i + self.lookback_period],
                features_scaled[i:i + self.lookback_period]
            )).reshape(self.lookback_period, 3)  # 3 features: value, day, hour
            X.append(sequence)
            y.append(data_scaled[i + self.lookback_period:i + self.lookback_period + self.prediction_window])
        return torch.FloatTensor(X), torch.FloatTensor(y)
    
    def train(self, historical_data, times, epochs=100, batch_size=32):
        if not isinstance(historical_data, np.ndarray):
            historical_data = np.array(historical_data)
        X, y = self.prepare_data(historical_data, times)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size].to(self.device)
                batch_y = y_train[i:i+batch_size].to(self.device)
                self.optimizer.zero_grad()
                output, _ = self.model.lstm(batch_X)
                output = self.model.fc(output[:, -1, :])
                loss = self.criterion(output, batch_y.squeeze())
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()
            
            self.model.eval()
            with torch.no_grad():
                val_output, _ = self.model.lstm(X_test.to(self.device))
                val_output = self.model.fc(val_output[:, -1, :])
                val_loss = self.criterion(val_output, y_test.squeeze().to(self.device))
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_train_loss/len(X_train):.6f}, Val Loss: {val_loss.item():.6f}")
        
        # the amonaly threshold based on percentile of errors
        self.model.eval()
        with torch.no_grad():
            predictions, _ = self.model.lstm(X.to(self.device))
            predictions = self.model.fc(predictions[:, -1, :])
            errors = torch.abs(predictions - y.to(self.device)).cpu().numpy().flatten()
            self.error_threshold = np.percentile(errors, self.anomaly_percentile)
            print(f"Error Threshold (95th percentile): {self.error_threshold:.4f}")
    
    def detect(self, new_value, time_step):
        self.history.append(new_value)
        self.time_points.append(time_step)
        self.values.append(new_value)
        
        if len(self.history) == self.lookback_period:
            current_sequence = np.array(self.history).reshape(-1, 1)
            scaled_sequence = self.scaler.transform(current_sequence)
            # Add day and hour features
            day = time_step % 7
            hour = (time_step * 24 % 24)
            features = np.array([[day, hour]] * self.lookback_period)
            features_scaled = self.feature_scaler.transform(features)
            model_input = np.hstack((scaled_sequence, features_scaled)).reshape(1, self.lookback_period, 3)
            model_input = torch.FloatTensor(model_input).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                output, _ = self.model.lstm(model_input)
                prediction_scaled = self.model.fc(output[:, -1, :])
                prediction = self.scaler.inverse_transform(prediction_scaled.cpu().numpy())[0][0]
            
            self.predictions.append(prediction)
            error = abs(new_value - prediction)
            is_anomaly = error > self.error_threshold * 1000  # Scale back to original units
            
            if is_anomaly:
                self.anomalies.append((time_step, new_value))
            else:
                self.anomalies.append((time_step, None))
                
            if time_step % 10 == 0:
                print(f"Time: {time_step}, Value: {new_value:.2f}, Pred: {prediction:.2f}, "
                      f"Error: {error:.2f}, Threshold: {self.error_threshold*1000:.2f}, Anomaly: {is_anomaly}")
                
            return (is_anomaly, new_value, prediction, self.error_threshold * 1000)
        self.predictions.append(None)
        return (False, new_value, None, None)

def animate(i, detector, ax):
    new_value = (np.sin(i * 2 * np.pi / 24) +  # Daily cycle
                 np.sin(i * 2 * np.pi / 168) +  # Weekly cycle
                 np.random.normal(0, 0.1)) * 1000  # Noise
    if i % 50 == 0:  # Introduce anomaly
        new_value *= 1.5
    
    is_anomaly, actual, predicted, threshold = detector.detect(new_value, i)
    
    ax.clear()
    ax.plot(detector.time_points, detector.values, 'b-', label='Actual')
    ax.plot(detector.time_points, detector.predictions, 'g--', label='Predicted')
    
    anomaly_points = [(t, v) for t, v in detector.anomalies if v is not None]
    if anomaly_points:
        anomaly_times, anomaly_values = zip(*anomaly_points)
        ax.scatter(anomaly_times, anomaly_values, color='r', label='Anomalies')
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Page Load Time (ms)')
    ax.set_title('Live Page Load Time Anomaly Detection')
    ax.legend()

def main():
    np.random.seed(42)
    t = np.linspace(0, 1000/24, 1000)  # Time in days for 1000 hourly points
    historical_data = (np.sin(t * 2 * np.pi) +  # Daily cycle
                      np.sin(t * 2 * np.pi / 7) +  # Weekly cycle
                      np.random.normal(0, 0.1, 1000)) * 1000  # Noise
    
    detector = TimeSeriesAnomalyDetector(anomaly_percentile=75)
    detector.train(historical_data, np.arange(1000) / 24)  # Pass time in days
    
    print("\nStarting live stream simulation with plotting...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ani = FuncAnimation(fig, animate, fargs=(detector, ax), interval=100, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    main()