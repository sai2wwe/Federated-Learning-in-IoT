import flwr as fl # type: ignore
import tensorflow as tf # type: ignore
import paho.mqtt.client as mqtt # type: ignore
import json
import time

# MQTT server setup
broker = "127.0.0.1"  # Replace with your broker address
server_client = mqtt.Client()

# Track federated learning round status
federated_round_in_progress = False

# Flower server setup
class FlowerServer(fl.server.Server):
    def __init__(self):
        super().__init__()

    def fit_round_started(self, round_num: int):
        print(f"Federated learning round {round_num} started.")
        global federated_round_in_progress
        federated_round_in_progress = True
        server_client.publish("server/start_federated_round")  # Notify clients to start federated learning

    def fit_round_ended(self, round_num: int, results):
        print(f"Federated learning round {round_num} ended.")
        global federated_round_in_progress
        federated_round_in_progress = False
        server_client.publish("server/end_federated_round")  # Notify clients to resume data collection

# Flower model configuration (for simplicity, using a basic model)
def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),  # Example input shape: temperature and humidity
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax")  # Output classes: 0 (safe), 1 (danger)
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Function to start federated learning
def start_federated_learning():
    print("Starting federated learning.")
    # Initialize Flower server with the model and training configurations
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=2,
        min_eval_clients=2,
        min_available_clients=2,
    )
    # Start Flower server
    fl.server.start_server("0.0.0.0:8080", strategy=strategy)

# MQTT callback functions for controlling federated rounds
def on_connect(client, userdata, flags, rc):
    print("Connected to broker.")
    client.subscribe("server/control")  # Listen for control commands (e.g., start or stop federated round)

def on_message(client, userdata, msg):
    global federated_round_in_progress
    if msg.topic == "server/control":
        control_message = json.loads(msg.payload.decode())
        if control_message.get("action") == "start":
            if not federated_round_in_progress:
                start_federated_learning()  # Trigger federated learning round
        elif control_message.get("action") == "stop":
            print("Stopping federated learning.")
            # You can add logic here to stop federated learning gracefully if needed

# Start the MQTT client loop for communication
def start_mqtt_server():
    server_client.on_connect = on_connect
    server_client.on_message = on_message
    server_client.connect(broker, 1883, 60)
    server_client.loop_start()

# Periodically check if federated round is in progress
def monitor_federated_round():
    while True:
        if federated_round_in_progress:
            print("Federated learning is in progress, waiting for updates.")
        else:
            print("Federated round completed. Waiting for new round trigger.")
        time.sleep(10)  # Monitor every 10 seconds

# Main entry point
def main():
    start_mqtt_server()
    monitor_federated_round()

if __name__ == "__main__":
    main()
