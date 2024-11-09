import argparse
import warnings
import pandas as pd
import flwr as fl
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

parser = argparse.ArgumentParser(description="Flower with MLP for Sensor Data")
parser.add_argument(
    "--server_address",
    type=str,
    default="0.0.0.0:8080",
    help="gRPC server address (default '0.0.0.0:8080')",
)
parser.add_argument(
    "--cid",
    type=int,
    required=True,
    help="Client ID. Should be an integer between 0 and NUM_CLIENTS",
)

warnings.filterwarnings("ignore", category=UserWarning)
NUM_CLIENTS = 50  # Adjust as needed for your simulation

def prepare_dataset(filepath: str):
    """Load and preprocess the DHT sensor dataset."""
    # Load the CSV file
    data = pd.read_csv(filepath)
    # Normalize sensor readings
    data = data.drop('timestamp', axis=1)
    data = (data - data.min()) / (data.max() - data.min())
    
    # Assuming 'label' is the target column and others are features
    features = data.drop(columns=['label']).values
    labels = data['label'].values
    
    # Split the dataset into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size=0.1, random_state=42)
    return (x_train, y_train), (x_val, y_val)

class FlowerClient(fl.client.NumPyClient):
    """A FlowerClient using a fully connected neural network (MLP) for sensor data classification."""

    def __init__(self, trainset, valset, input_shape, num_classes):
        self.x_train, self.y_train = trainset
        self.x_val, self.y_val = valset
        # Initialize a simple fully connected neural network
        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=input_shape),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )
        # Compile the model
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, params):
        self.model.set_weights(params)

    def fit(self, parameters, config):
        print("Client sampled for fit()")
        # Set model weights and hyperparameters
        self.set_parameters(parameters)
        batch_size = config.get("batch_size", 32)
        epochs = config.get("epochs", 5)
        # Train the model
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size)
        return self.get_parameters({}), len(self.x_train), {}

    def evaluate(self, parameters, config):
        print("Client sampled for evaluate()")
        self.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(self.x_val, self.y_val)
        return loss, len(self.x_val), {"accuracy": accuracy}

def main():
    args = parser.parse_args()
    # Prepare dataset (load and split into train and validation sets)
    trainset, valset = prepare_dataset("dht_readings.csv")

    # Define input shape and number of classes
    input_shape = trainset[0].shape[1:]
    num_classes = len(set(trainset[1]))

    # Start Flower client with MLP model
    fl.client.start_client(
        server_address=args.server_address,
        client=FlowerClient(
            trainset=trainset, valset=valset, input_shape=input_shape, num_classes=num_classes
        ),
    )

if __name__ == "__main__":
    main()
