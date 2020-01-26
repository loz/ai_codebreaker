from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import codebreaker as cb
import numpy as np


class WindowTensorPlayer:
  def __init__(self):
    self.window = self.gen_window()

  def guess(self, pegs, clue):
    self.window = self.window_move(self.window, pegs, clue)
    state = self.flatten_window(self.window)
    choice = self.model.predict([state])
    choice = tf.reshape(choice, (6,4))
    choice = tf.math.argmax(choice)
    choice = choice.numpy().tolist()
    guess = []
    for peg in choice:
      guess.append(peg+1)
    return guess

  def build_model(self):
    self.model = tf.keras.models.Sequential([
              tf.keras.layers.Dense(8*5, input_shape=(8*5,)),
              tf.keras.layers.Dense(256, activation='relu'),
              tf.keras.layers.Dense(128, activation='relu'),
              tf.keras.layers.Dense(64, activation='relu'),
              tf.keras.layers.Dropout(0.2),
              tf.keras.layers.Dense(4*6, activation='relu')
            ])
    self.model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

  def train_model(self, training_data, epochs=5):
    x_train, y_train = self.reshape_training_data(training_data)
    self.model.fit(x_train, y_train, epochs=epochs)

  def gen_window(self):
    return [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
      ]

  def window_move(self, window, guess, clue):
    window = window[1:5]
    window.append(guess + clue)
    return window
    

  def reshape_training_data(self, training_data):
    """
  WANT:

  Lookbck 5 goes
  0: [0000] [0000] -> [1111]
  1: [1111] [????] -> [2222]
  2: [2222] [????] -> [3333]
  3: [3333] [????] -> [4444]
  4: [4444] [????] -> [5555]

  NEED:
    Reshape game runs to have 5 step window
    Tell NN that score wants to be [2 2 2 2]


    """
    x_train = []
    y_train = []
    for game in training_data:
      window = self.gen_window()
      for move in game:
        window = self.window_move(window, move['prior_guess'], move['prior_clue'])
        x_train.append(self.flatten_window(window))
        y_train.append(self.reshape_guess(move["guess"]))
    return x_train, y_train

  def flatten_window(self, window):
    flat = []
    for row in window:
      flat += row
    return flat

  def reshape_guess(self, guess):
    target = []
    for peg in guess:
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
    return rtarget

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
              tf.keras.layers.Dense(256, activation='relu'),
              tf.keras.layers.Dense(128, activation='relu'),
              tf.keras.layers.Dense(64, activation='relu'),
              tf.keras.layers.Dropout(0.1),
              tf.keras.layers.Dense(4*6, activation='relu')
            ])
    self.model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

  def train_model(self, training_data, epochs=15):
    x_train, y_train = self.reshape_training_data(training_data)
    self.model.fit(x_train, y_train, epochs=epochs)

  def reshape_training_data(self,training_data):
    """
    training_data = [
      [{guess1},{guess2}..]
    ]
    """
  
    x_train = []
    y_train = []
    for game in training_data:
      for guess in game:
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


def tensor_game(player):
  game = cb.Game(player, render=True)
  return game.play(20)

#player = TensorPlayer()
player = WindowTensorPlayer()
#training_data = cb.generate_random_guesses(10)
training_data = cb.generate_optimal_guesses(100)
player.build_model()
player.train_model(training_data, 5)
tensor_game(player)
