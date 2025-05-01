import numpy as np

from pyinsurance.portfolio.tipp import TIPP

tipp = TIPP(
    capital=100,
    multiplier=2,
    rr=np.array([0.1, -0.5, 0.89, 0.1]),
    rf=np.array([0.001, 0.002, 0.001, 0.001]),
    lock_in=0.05,
    min_risk_req=0.80,
    min_capital_req=0.80,
)

tipp.run()
