from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import codebreaker as cb
import numpy as np


class TensorPlayer:
  def guess(self, pegs, clue):
    choice = self.model.predict([pegs + clue])
    choice = tf.reshape(choice, (6,4))
    choice = tf.math.argmax(choice)
    choice = choice.numpy().tolist()
    guess = []
    for peg in choice:
      guess.append(peg+1)
    return guess

  def build_model(self):
    self.model = tf.keras.models.Sequential([
              tf.keras.layers.Dense(8, input_shape=(8,)),
              tf.keras.layers.Dense(128, activation='relu'),
              tf.keras.layers.Dropout(0.2),
              tf.keras.layers.Dense(4*6, activation='relu'),
            ])
    self.model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

  def train_model(self, x_train, t_train, epochs=15):
    self.model.fit(x_train, y_train, epochs=epochs)


def build_training_data(size = 2000):
  training_data = cb.generate_random_guesses(size)
  #training_data = cb.generate_optimal_guesses(size)
  
  x_train = []
  y_train = []
  for guess in training_data:
    x_train.append(guess["prior_guess"] + guess["prior_clue"])
    target = []
    for peg in guess["guess"]:
      v = [0, 0, 0, 0, 0, 0]
      v[peg-1] = 1
      target.append(v)
  
    #Rotate Orientation
    rtarget = []
    for i in range(0,6):
      rtarget.append(target[0][i])
      rtarget.append(target[1][i])
      rtarget.append(target[2][i])
      rtarget.append(target[3][i])
  
    y_train.append(rtarget)
  return x_train, y_train


"""
guess = [0, 0, 0, 0]
clue = [0, 0, 0, 0]

game = cb.Codebreaker()
game.random_code()

game.render(guess)
game.render_clue(clue)
print()
"""

"""
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
print("====ACUTAL====")
game.render()
print
"""

def tensor_game(player):
  game = cb.Game(player, render=True)
  return game.play(20)

player = TensorPlayer()
x_train, y_train = build_training_data()
player.build_model()
player.train_model(x_train, y_train)
tensor_game(player)
