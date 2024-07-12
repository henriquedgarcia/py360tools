import unittest

from test.test_cmp.test_cmp import TestCmp
from test.test_cmp.test_cmp_transform import TestCmpTransform
from test.test_cmp.test_draw_cmp import TestCmpDrawMethods
from test.test_erp.test_draw_erp import TestErpDrawMethods
from test.test_erp.test_erp import TestErp
from test.test_erp.test_erp_transform import TestErpTransform
from test.test_utils.test_hm import TestPosition2Trajectory
from test.test_viewport.test_viewport import TestViewport


def create_test_suite():
    suite = unittest.TestSuite()

    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestPosition2Trajectory))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestViewport))

    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestCmpTransform))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestCmp))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestCmpDrawMethods))

    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestErpTransform))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestErp))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestErpDrawMethods))

    return suite


if __name__ == '__main__':
    test_suite = create_test_suite()
    runner = unittest.TextTestRunner()
    runner.run(test_suite)
