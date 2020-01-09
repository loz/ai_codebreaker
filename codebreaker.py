import copy
import sys
import random

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
    keys = []
    code = copy.copy(self.code)
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


def random_game():
  game = Codebreaker()
  game.random_code()
  game.render()
  print
  print "-------- START STATE ------"
  game.render([0,0,0,0])
  game.render_clue([0,0,0,0])
  print
  print "-------- GUESSING ------"
  for i in range(0,20):
    guess = []
    for i in range(0,4):
      guess.append(random.randrange(0,6)+1)
    keys = game.clue(guess)
    game.render(guess)
    game.render_clue(keys)
    print

if __name__ == '__main__':
  random_game()
