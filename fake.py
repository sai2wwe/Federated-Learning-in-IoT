import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('modifi.csv')
print(data.head())
data = data.drop('timestamp', axis=1)
print(data.max(), data.min())
data = (data - data.min()) / (data.max() - data.min())
    
#     # Assuming 'label' is the target column and others are features
# features = data.drop(columns=['label']).values
# labels = data['label'].values
    
#     # Split the dataset into training and validation sets
# x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size=0.1, random_state=42)
# trainset,valset = (x_train, y_train), (x_val, y_val)
# input_shape = trainset[0].shape[1:]

# model = tf.keras.Sequential(
#             [
#                 tf.keras.layers.Input(shape=input_shape),
#                 tf.keras.layers.Dense(64, activation="relu"),
#                 tf.keras.layers.Dense(32, activation="relu"),
#                 tf.keras.layers.Dense(2, activation="softmax"),
#             ]
#         )
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# model.fit(x_train, y_train, epochs=5)

# loss, accuracy = model.evaluate(x_val, y_val)

# print(loss, accuracy)
# model.save('fake_model.h5')