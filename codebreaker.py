import copy

class Codebreaker:
  def __init__(self):
    self.code = []

  def clue(self, guess):
    keys = []
    code = copy.copy(self.code)
    for idx in range(0,4):
      peg = guess[idx]
      if self.code[idx] == peg:
        keys.append(2)
      elif peg in code:
        keys.append(1)
        code.remove(peg)
    while len(keys) < 4:
      keys.append(0)
    keys.sort(reverse=True)
    return keys
