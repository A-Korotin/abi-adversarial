import unittest
import numpy as np
from abi.calibration.moments import vacancy_spells, kaplan_meier, km_median


class MyTestCase(unittest.TestCase):
    def test_moments(self):
        occ = np.array([[True], [False], [False], [True], [False], [False]])
        d, obs = vacancy_spells(occ, burn_in=0)
        self.assertListEqual(list(d), [2, 2])
        self.assertListEqual(list(obs), [True, False])

        times, S = kaplan_meier(d, obs)
        self.assertListEqual(list(times),[2])
        self.assertTrue(np.isclose(S[0], 0.5))
        self.assertEqual(km_median(times, S), 2)


if __name__ == '__main__':
    unittest.main()
