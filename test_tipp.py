import numpy as np
from pyinsurance.portfolio.tipp__ import TIPP


def main():
    # Generate sample data
    n_periods = 252  # One year of daily data
    np.random.seed(42)

    # Generate returns - ensure they are float64 arrays
    rr = np.array(
        np.random.normal(0.0005, 0.02, n_periods), dtype=np.float64
    )  # Risky asset returns
    rf = np.array(np.ones(n_periods) * 0.0001, dtype=np.float64)  # Risk-free returns
    br = np.array(
        np.random.normal(0.0003, 0.015, n_periods), dtype=np.float64
    )  # Benchmark returns

    # Reshape to (1, N) as required by TIPP
    rr = rr.reshape(1, -1)
    rf = rf.reshape(1, -1)
    br = br.reshape(1, -1)

    # Initialize TIPP strategy
    tipp = TIPP(
        capital=1000000.0,  # Initial capital
        floor=800000.0,  # Initial floor
        multiplier=2.0,  # Risk multiplier
        rr=rr,  # Risky returns
        rf=rf,  # Risk-free returns
        br=br,  # Benchmark returns
        lock_in=0.1,  # 10% lock-in
        min_risk_req=0.2,  # Minimum 20% in risky assets
        min_capital_req=0.8,  # Minimum 80% of reference capital
    )

    # Run the strategy
    tipp.run()

    # Access results through properties
    print(f"Final portfolio value: {tipp.portfolio[-1]:.2f}")
    print(f"Final reference capital: {tipp.reference_capital[-1]:.2f}")
    print(f"Total margin injections: {tipp.margin_trigger.sum():.2f}")

    # Calculate some statistics
    returns = np.diff(tipp.portfolio) / tipp.portfolio[:-1]
    print(f"Annualized return: {(1 + returns.mean())**252 - 1:.2%}")
    print(f"Annualized volatility: {returns.std() * np.sqrt(252):.2%}")


if __name__ == "__main__":
    main()
