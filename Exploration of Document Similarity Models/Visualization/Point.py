

class Point(object):
    """
    Component to specify a 2-D point
    """
    def __init__(self, x: float, y:float):
        """
        Constructor
        :param x:
        :param y:
        """
        self.x = x
        self.y = y

    def __str__(self):
        """
        String representation of the point
        :return:
        """
        msg = f"(x, y) = ({self.x}, {self.y})"
        return msg