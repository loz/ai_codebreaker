from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import codebreaker as cb

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(8, input_shape=(8,)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(4*6, output_shape=(4,6), activation='softmax')
])

model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

training_data = cb.generate_random_guesses(100)
x_train = []
y_train = []
for guess in training_data:
  x_train.append(guess["prior_guess"] + guess["prior_clue"])
  y_train.append(guess["guess"])

model.fit(x_train, y_train, epochs=5)
print(model.predict([[1,2,3,4,2,1,0,0]], verbose=2))
