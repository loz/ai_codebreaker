import copy
import sys
import random

import tensorflow as tf
from tf_agents.trajectories import time_step
from tf_agents import specs
from tf_agents.environments import py_environment
import numpy as np

class RLEnv(py_environment.PyEnvironment):

  def __init__(self):
    self.guesses = 0
    self.game = Game(RandomPlayer())

  def _reset(self):
    self.guesses = 0
    self.game = Game(RandomPlayer())
    return self.step_data()

  def _step(self, action):
    self.guesses += 1
    self.game.guess(action)
    return self.step_data()

  def step_data(self):
    clue = np.array(self.game.prior_clue)
    clue = clue * clue
    #reward = np.array([np.sum(clue)])
    reward = np.sum(clue)
    observation = tf.convert_to_tensor(np.array([self.game.prior_guess, self.game.prior_clue]))
    if self.guesses > 10:
      return time_step.termination(observation, reward)
    else:
      return time_step.transition(observation, reward)

  def render(self):
    self.game.render()

  def action_spec(self):
    return specs.array_spec.BoundedArraySpec(
      (4,), np.int64,
      minimum=1,
      maximum=6,
      name='action'
    )

  def observation_spec(self):
    return specs.array_spec.BoundedArraySpec(
      (2,4), np.int64,
      minimum=0,
      maximum=6,
      name='observation')


class Codebreaker:

  def __init__(self):
    self.code = []
    self.keycolors = {
      0 : "\033[49m",
      1 : "\033[47m",
      2 : "\033[100m"
    }
    self.colors = {
      0 : "\033[49m",
      1 : "\033[41m",
      2 : "\033[42m",
      3 : "\033[43m",
      4 : "\033[44m",
      5 : "\033[45m",
      6 : "\033[46m",
    }
    self.reset = "\033[49m"

  def random_code(self):
    random.seed
    for i in range(0,4):
      self.code.append(random.randrange(0,5) + 1)

  def render(self, pegs = None):
    if pegs == None:
      pegs = self.code

    for peg in pegs:
      sys.stdout.write(self.colors[peg])
      sys.stdout.write("%d" % peg)
      sys.stdout.write(self.reset)
      sys.stdout.write(" ")

  def render_clue(self, keys):
    sys.stdout.write("  [ ")
    for key in keys:
      sys.stdout.write(self.keycolors[key])
      sys.stdout.write("%d" % key)
      sys.stdout.write(self.reset)
      sys.stdout.write(" ")
    sys.stdout.write("]")
    pass

  def clue(self, guess):
    print(guess)
    keys = []
    code = copy.copy(self.code)
    if hasattr(guess, 'tolist'):
      cguess = guess.tolist()
    else:
      cguess = copy.copy(guess)

    #first do exact matches
    for idx in range(0,4):
      peg = guess[idx]
      if self.code[idx] == peg:
        keys.append(2)
        cguess.remove(peg)
        code.remove(peg)

    for peg in cguess:
      if peg in code:
        keys.append(1)
        code.remove(peg)

    while len(keys) < 4:
      keys.append(0)
    keys.sort(reverse=True)
    #self.render(guess)
    #self.render_clue(keys)
    #print
    return keys


class RandomPlayer:
  def guess(self, pegs, clue):
    guess = []
    for i in range(0,4):
      guess.append(random.randrange(0,6)+1)
    return guess

class OptimalPlayer:
  def __init__(self):
    self.refboard = Codebreaker()
    self.options = []
    for first in range(1,7):
      for second in range(1,7):
        for third in range(1,7):
          for fourth in range(1,7):
            self.options.append([first, second, third, fourth])
    self.firstguess = True

  def guess(self, pegs, clue):
    if self.firstguess:
      self.firstguess = False 
      return [1, 1, 2, 2] #Perfect First Guess
    else:
      self.filter_options(pegs, clue)
      return random.choice(self.options)

  def test_option(self, option, pegs, clue):
    self.refboard.code = option
    return self.refboard.clue(pegs) == clue
  
  def filter_options(self, pegs, clue):
    self.options = [opt for opt in self.options if self.test_option(opt, pegs, clue)]

class Game:

  def __init__(self, player, render=False):
    self.guesses = []
    self.game = Codebreaker()
    self.game.random_code()
    self.prior_guess = [0,0,0,0]
    self.prior_clue = [0,0,0,0]
    self.lastscore = 0
    self.rendering = render
    self.player = player
    if self.rendering:
      print()
      print("-------- START STATE ------")
      self.game.render(self.prior_guess)
      self.game.render_clue(self.prior_clue)
      print()
      print("-------- GUESSING ------")

  def render(self):
    self.game.render(self.prior_guess)
    self.game.render_clue(self.prior_clue)
    print()

  def guess(self, guess):
    keys = self.game.clue(guess)
    self.guesses.append({
      "prior_guess" : self.prior_guess,
      "prior_clue" : self.prior_clue,
      "guess" : guess
      })
    self.lastscore = sum(keys)
    self.prior_guess = guess
    self.prior_clue = keys
    if self.rendering:
      self.render()
    return keys == [2, 2, 2, 2] #WON
    

  def round(self):
    guess = self.player.guess(self.prior_guess, self.prior_clue)
    return self.guess(guess)

  def play(self, n=15):
    for i in range(0,n):
      over = self.round()
      if over:
        break
    if (over and self.render):
      print("----- WON -----")
    if self.render:
      print("===== TARGET =====")
      self.game.render()
      print()
    return self.guesses

def random_game():
  game = Game(RandomPlayer())
  return game.play(20)

def optimal_game():
  game = Game(OptimalPlayer())
  return game.play(20)

def generate_random_guesses(n=500):
  sys.stdout.write("Generating Random\n")
  games = []
  for run in range(1,n):
    sys.stdout.write(".")
    sys.stdout.flush()
    games.append(random_game())
  return games

def generate_optimal_guesses(n=500):
  sys.stdout.write("Generating Optimal\n")
  games = []
  for run in range(1,n):
    sys.stdout.write(".")
    sys.stdout.flush()
    games.append(optimal_game())
  return games

if __name__ == '__main__':
  generate_optimal_guesses()
