import unittest

from test.test_cmp.test_cmp import TestCmp
from test.test_erp.test_erp import TestErp
from test.test_utils.test_hm import TestPosition2Trajectory


def create_test_suite():
    suite = unittest.TestSuite()

    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestPosition2Trajectory))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestCmp))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestErp))

    return suite


if __name__ == '__main__':
    test_suite = create_test_suite()

    runner = unittest.TextTestRunner()
    runner.run(test_suite)
