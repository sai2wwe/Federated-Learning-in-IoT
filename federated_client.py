import argparse
import warnings
import pandas as pd # type: ignore
import flwr as fl # type: ignore
import tensorflow as tf # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from tensorflow import keras # type: ignore
import time
import paho.mqtt.client as mqtt # type: ignore
import json

# Define thresholds for temperature and humidity
TEMPERATURE_THRESHOLD = 30  # Example threshold
HUMIDITY_THRESHOLD = 70     # Example threshold

# MQTT client setup
broker = "127.0.0.1"  # replace with your broker address
client = mqtt.Client()

# Flag to control sensor data collection during federated learning
collecting_data = True

# Function to collect real-time sensor data
def collect_sensor_data():
    if collecting_data:
        # Simulate real-time sensor data collection
        temperature = 23 + (time.time() % 10)  # Example temp reading
        humidity = 55 + (time.time() % 10)  # Example humidity reading
        return temperature, humidity
    return None, None

# Function to update CSV with real-time sensor data and threshold-based labels
def update_csv_with_sensor_data(temperature, humidity):
    # Create or append data to CSV with real-time readings and thresholds
    new_data = pd.DataFrame({
        'temperature': [temperature],
        'humidity': [humidity],
        'timestamp': [time.strftime('%Y-%m-%d %H:%M:%S')],
        'label': [1 if temperature > TEMPERATURE_THRESHOLD or humidity > HUMIDITY_THRESHOLD else 0]
    })

    try:
        # If CSV file exists, append to it
        data = pd.read_csv('dht_readings.csv')
        data = pd.concat([data, new_data], ignore_index=True)
    except FileNotFoundError:
        # If CSV file does not exist, create a new one
        data = new_data

    data.to_csv("dht_readings.csv", index=False)  # Save updated data

# Prepare dataset for model training
def prepare_dataset(filepath: str):
    """Load, preprocess, and apply threshold labeling to the DHT sensor dataset."""
    data = pd.read_csv(filepath)
    data['label'] = ((data['temperature'] > TEMPERATURE_THRESHOLD) |
                     (data['humidity'] > HUMIDITY_THRESHOLD)).astype(int)

    # Drop timestamp and normalize other features
    data = data.drop('timestamp', axis=1)
    features = data.drop(columns=['label']).values
    labels = data['label'].values

    # Split dataset into training and validation
    x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size=0.1, random_state=42)
    return (x_train, y_train), (x_val, y_val)

class FlowerClient(fl.client.NumPyClient):
    """A FlowerClient using a fully connected neural network (MLP) for sensor data classification."""
    def __init__(self, trainset, valset, input_shape, num_classes):
        self.x_train, self.y_train = trainset
        self.x_val, self.y_val = valset
        # Define a simple fully connected neural network model
        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=input_shape),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, params):
        self.model.set_weights(params)

    def fit(self, parameters, config):
        print("Client sampled for fit()")
        self.set_parameters(parameters)
        batch_size = config.get("batch_size", 32)
        epochs = config.get("epochs", 5)
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size)
        return self.get_parameters({}), len(self.x_train), {}

    def evaluate(self, parameters, config):
        print("Client sampled for evaluate()")
        self.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(self.x_val, self.y_val)
        return loss, len(self.x_val), {"accuracy": accuracy}

# MQTT callback functions for server communication
def on_connect(client, userdata, flags, rc):
    print("Connected to broker.")
    client.subscribe("server/start_federated_round")

def on_message(client, userdata, msg):
    global collecting_data
    if msg.topic == "server/start_federated_round":
        print("Federated round started, pausing real-time sensor collection.")
        collecting_data = False  # Stop collecting data during training
        client.unsubscribe("sensor/data")  # Stop receiving real-time data
        federated_learning_round()
    elif msg.topic == "server/end_federated_round":
        print("Federated round ended, resuming sensor collection.")
        collecting_data = True  # Resume collecting data
        client.subscribe("sensor/data")

# Federated learning round
def federated_learning_round():
    print("Federated learning round in progress.")
    # Prepare dataset for training
    trainset, valset = prepare_dataset("dht_readings.csv")

    # Initialize Flower client and start training
    input_shape = trainset[0].shape[1:]
    num_classes = len(set(trainset[1]))
    fl.client.start_client(
        server_address="0.0.0.0:8080",  # Server address
        client=FlowerClient(trainset, valset, input_shape, num_classes)
    )

# Start the MQTT client loop
def start_mqtt_client():
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker, 1883, 60)
    client.loop_start()

# Function to send real-time data to server (in case the client is allowed to send)
def send_real_time_data():
    while collecting_data:
        temperature, humidity = collect_sensor_data()
        if temperature is not None and humidity is not None:
            update_csv_with_sensor_data(temperature, humidity)  # Update the CSV with the new sensor data
            sensor_data = {"temperature": temperature, "humidity": humidity}
            client.publish("sensor/data", json.dumps(sensor_data))  # Send data to server
        time.sleep(5)  # Delay between readings

def main():
    start_mqtt_client()
    send_real_time_data()

if __name__ == "__main__":
    main()
