import unittest as ut
import inspect

import Visualization.Point as p


class TestPoint(ut.TestCase):
    """
    Test suit for 'Point' class
    """
    def test_Point_Construction_Is_Valid(self):
        """
        test the validity of 'Point' constructor
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        point = p.Point(x=10, y=20)
        self.assertIsNotNone(point, msg=error_msg)
        print(f"Point is: {point}")


if __name__ == '__main__':
    ut.main()
