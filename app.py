from flask import Flask, jsonify, render_template
import tensorflow as tf
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the machine learning model
model = tf.keras.models.load_model('fake_model.h5')

# Load the CSV file and set up a data iterator
data = pd.read_csv('dht_readings.csv')
data_iterator = iter(data.iterrows())  # Iterator to read each row

def predict_label(temperature, humidity):
    # Prepare the data for the model (reshape to match input)
    input_data = np.array([[temperature, humidity]])
    prediction = model.predict(input_data)
    print(prediction)
    return bool(prediction[0][0] > 0.5)  # Using 0.5 as the threshold for True/False

def get_next_reading():
    global data_iterator
    try:
        # Get the next row of data
        _, row = next(data_iterator)
        temperature = row['temperature']
        humidity = row['humidity']
        print(f"Reading - Temperature: {temperature}, Humidity: {humidity}")
        return {'temperature': temperature, 'humidity': humidity}
    except StopIteration:
        # Restart the iterator if we reach the end of the data
        print("Reached end of CSV, restarting iterator.")
        data_iterator = iter(data.iterrows())
        _, row = next(data_iterator)
        return {'temperature': row['temperature'], 'humidity': row['humidity']}
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    # Get the next reading from the CSV data

    # temp, humd = 20.37836619715026,44.81801410483999
    max_temp, max_humd = 30.7, 92.6
    min_temp, min_humd = 20.004354, 30.009223
    scaled_temp = (temp - min_temp) / (max_temp - min_temp)
    scaled_humd = (humd - min_humd )/ (max_humd - min_humd)
    sensor_data = {'temperature': scaled_temp, 'humidity': scaled_humd}
    print(f"Sensor Data: {sensor_data}")

    # Ensure data is valid
    if sensor_data['temperature'] is None or sensor_data['humidity'] is None:
        return jsonify({'label': 'Invalid data'})

    # Predict label using model
    label = predict_label(sensor_data['temperature'], sensor_data['humidity'])
    print(f"Prediction: {label}")
    return jsonify({'label': label, 'temperature': sensor_data['temperature'], 'humidity': sensor_data['humidity']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
