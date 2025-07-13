import unittest

from tests.test_cmp.test_cmp import TestCmp
from tests.test_cmp.test_cmp_transform import TestCmpTransform
from tests.test_cmp.test_draw_cmp import TestCmpDrawMethods
from tests.test_erp.test_draw_erp import TestErpDrawMethods
from tests.test_erp.test_erp import TestErp
from tests.test_erp.test_erp_transform import TestErpTransform
from tests.test_utils.test_hm import TestPosition2Trajectory
from tests.test_viewport.test_viewport import TestViewport

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)

    print('\nRUNNING VIEWPORT TESTS')
    suite_viewport = unittest.TestSuite()
    suite_viewport.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestPosition2Trajectory))
    suite_viewport.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestViewport))
    runner.run(suite_viewport)

    print('\nRUNNING CMP TESTS')
    suite_cmp = unittest.TestSuite()
    suite_cmp.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestCmpTransform))
    suite_cmp.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestCmp))
    suite_cmp.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestCmpDrawMethods))
    runner.run(suite_cmp)

    print('\nRUNNING ERP TESTS')
    suite_erp = unittest.TestSuite()
    suite_erp.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestErpTransform))
    suite_erp.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestErp))
    suite_erp.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestErpDrawMethods))
    runner.run(suite_erp)
