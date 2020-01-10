from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import codebreaker as cb

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(8, input_shape=(8,)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(4*6, activation='relu'),
])

model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

training_data = cb.generate_random_guesses(1000)
x_train = []
y_train = []
for guess in training_data:
  x_train.append(guess["prior_guess"] + guess["prior_clue"])
  target = []
  for peg in guess["guess"]:
    v = [0, 0, 0, 0, 0, 0]
    v[peg-1] = 1
    target += v
  y_train.append(target)

model.fit(x_train, y_train, epochs=10)

guess = [0, 0, 0, 0]
clue = [0, 0, 0, 0]

game = cb.Codebreaker()
game.random_code()

game.render(guess)
game.render_clue(clue)
print()

for turn in range(1,12):
  choice = model.predict([guess + clue])
  choice = tf.reshape(choice, (6,4))
  choice = tf.math.argmax(choice)
  choice = choice.numpy().tolist()
  guess = []
  for peg in choice:
    guess.append(peg+1)
  clue = game.clue(guess)
  game.render(guess)
  game.render_clue(clue)
  print()
