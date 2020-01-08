import unittest
import codebreaker as cb

class TestCodeBreaker(unittest.TestCase):

  def __init__(self, args):
    super(TestCodeBreaker, self).__init__(args)
    self.game = cb.Codebreaker()
  
  def test_none_right_no_marks(self):
    self.game.code = [0,1,2,3]
    self.assertEqual(self.game.clue([4,4,4,4]), [0,0,0,0])

  def test_right_wrong_place_gives_1_mark(self):
    self.game.code = [0,1,2,3]
    self.assertEqual(self.game.clue([4,4,3,4]), [1,0,0,0])

  def test_right_wrong_twice_gives_1_mark(self):
    self.game.code = [0,1,2,3]
    self.assertEqual(self.game.clue([4,3,3,4]), [1,0,0,0])

  def test_two_right_twice_gives_two_1_marks(self):
    self.game.code = [0,2,2,3]
    self.assertEqual(self.game.clue([2,4,4,2]), [1,1,0,0])

  def test_one_right_in_right_place_gives_2_mark(self):
    self.game.code = [0,1,2,3]
    self.assertEqual(self.game.clue([0,4,4,4]), [2,0,0,0])

  def test_mixed_guess_gives_ordered_marks(self):
    self.game.code = [0,1,2,3]
    self.assertEqual(self.game.clue([3,1,2,0]), [2,2,1,1])


if __name__ == '__main__':
  unittest.main()
