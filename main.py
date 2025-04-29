import numpy as np
from pyinsurance.portfolio import TIPP
import time

# Example usage
capital = 1000000.0
floor = 800000.0
multiplier = 2.0

# Create sample rate arrays (1 row, N columns)
n_periods = 10000000  # e.g., daily data for a year
rr = np.random.normal(0, 0, (n_periods,))  # risky returns
rf = np.full((n_periods,), 0.00000000000001)  # risk-free returns
br = np.random.normal(0, 0, (n_periods,))  # benchmark returns

# Create TIPP instance
tipp = TIPP(
    capital=capital,
    multiplier=multiplier,
    rr=rr,
    rf=rf,
    lock_in=0.1,  # 10% lock-in
    min_risk_req=0.2,  # minimum 20% in risky assets
    min_capital_req=0.8,  # maintain 80% of capital
    freq=252,  # daily frequency
)

# Run the simulation with timing
start_time = time.time()
tipp.run()
end_time = time.time()

# Print timing results
print(f"Total execution time: {end_time - start_time:.4f} seconds")
print(
    f"Time per iteration: {(end_time - start_time) / n_periods * 1e6:.4f} microseconds"
)

# Access results
print("Final portfolio value:", tipp.portfolio[-1])
print("Portfolio values over time:", tipp.portfolio)
print("Reference capital:", tipp.ref_capital)
print("Margin triggers:", tipp.margin_trigger)
