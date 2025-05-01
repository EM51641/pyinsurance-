import numpy as np

from pyinsurance.portfolio.tipp import TIPP as PythonTIPP


class TestTIPP:

    def test_initialization(self):
        tipp = PythonTIPP(
            capital=100,
            multiplier=2,
            rr=np.array([0.1, -0.5, 0.89, 0.1]),
            rf=np.array([0.001, 0.002, 0.001, 0.001]),
            lock_in=0.05,
            min_risk_req=0.80,
            min_capital_req=0.80,
        )
        assert tipp.capital == 100
        assert tipp.multiplier == 2
        assert np.array_equal(tipp.rr, np.array([0.1, -0.5, 0.89, 0.1]))
        assert np.array_equal(tipp.rf, np.array([0.001, 0.002, 0.001, 0.001]))
        assert tipp.lock_in == 0.05
        assert tipp.min_risk_req == 0.80
        assert tipp.min_capital_req == 0.80

        assert tipp.portfolio is None
        assert tipp.ref_capital is None
        assert tipp.margin_trigger is None
        assert tipp.floor is None

    def test_tipp_runs(self):
        tipp = PythonTIPP(
            capital=100,
            multiplier=2,
            rr=np.array([0.1, -0.5, 0.89, 0.1]),
            rf=np.array([0.001, 0.002, 0.001, 0.001]),
            lock_in=0.05,
            min_risk_req=0.80,
            min_capital_req=0.80,
        )

        tipp.run()
        # Verify arrays are initialized
        assert tipp.portfolio is not None
        assert tipp.ref_capital is not None
        assert tipp.margin_trigger is not None
        assert tipp.floor is not None

        # Verify array lengths
        assert len(tipp.portfolio) == len(tipp.rr)
        assert len(tipp.ref_capital) == len(tipp.rr)
        assert len(tipp.margin_trigger) == len(tipp.rr)
        assert len(tipp.floor) == len(tipp.rr)

    def test_tipp_runs_correct_floor(self):
        rr = np.array([0.5, -0.5])
        rf = np.array([0.001, 0.001])
        lock_in = 0.05
        min_capital_req = 0.80
        min_risk_req = 0.80
        tipp = PythonTIPP(
            capital=100,
            multiplier=2,
            rr=rr,
            rf=rf,
            lock_in=lock_in,
            min_capital_req=min_capital_req,
            min_risk_req=min_risk_req,
        )
        tipp.run()

        discount_t0 = (1 + rf[0]) ** (2 / 252)
        discount_t1 = (1 + rf[1]) ** (1 / 252)

        assert tipp.floor[0] == 100 * min_capital_req * 1 / discount_t0
        assert tipp.floor[1] == tipp.portfolio[0] * min_capital_req * 1 / discount_t1
