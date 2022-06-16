import unittest as ut
import inspect
import numpy as np
from pprint import pprint

import Visualization.FeatureMatrix as f


class TestFeatureMatrixData(ut.TestCase):
    """
    Test suit for 'FeatureMatrixData' class
    """
    def test_FeatureMatrixData_Construction_Is_Valid(self):
        """
        test the validity of 'FeatureMatrixData' constructor
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        np.random.seed(1)
        data_2d = np.random.randint(10, size=(20, 2))
        doc_ids = list(range(20))
        feature_data = f.FeatureMatrixData(data_2d, doc_ids)
        self.assertIsNotNone(feature_data, msg=error_msg)
        pprint(f"Feature_data: {feature_data}")


if __name__ == '__main__':
    ut.main()
