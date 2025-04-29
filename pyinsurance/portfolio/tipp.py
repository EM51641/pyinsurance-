from typing import Optional
import numpy as np
from numpy.typing import NDArray

OPEN_DAYS_PER_YEAR = 252.0


class TIPP:
    """Time Invariant Portfolio Protection (TIPP) implementation.

    This class implements the core TIPP strategy logic, managing the dynamic
    allocation between risky and safe assets to protect against downside risk
    while maintaining upside potential.

    Attributes:
        capital: Initial investment capital
        multiplier: Risk multiplier that determines the aggressiveness of the strategy
        rr: Return rate of the risky asset
        rf: Risk-free rate of return
        lock_in: Lock-in percentage for gains (0-1)
        min_risk_req: Minimum risk requirement for the portfolio
        min_capital_req: Minimum capital requirement for the portfolio
        freq: Number of trading days per year
        portfolio: Portfolio value at each time step
        ref_capital: Reference capital at each time step
        margin_trigger: Margin trigger at each time step
        floor: Floor at each time step
        compounded_period: Compounded period
        discount: Discount at each time step
    """

    def __init__(
        self,
        capital: float,
        multiplier: float,
        rr: NDArray[np.float64],
        rf: NDArray[np.float64],
        lock_in: float,
        min_risk_req: float,
        min_capital_req: float,
        freq: float = 252,
    ) -> None:
        """
        Initialize TIPP model with parameters.

        Args:
            capital (float): Initial investment capital
            floor (float): Minimum acceptable portfolio value (protection level)
            multiplier (float): Risk multiplier that determines the aggressiveness of the strategy
            rr (np.ndarray): Return rate of the risky asset
            rf (np.ndarray): Risk-free rate of return
            br (np.ndarray): Benchmark return rate
            lock_in (float): Lock-in percentage for gains (0-1)
            min_risk_req (float): Minimum risk requirement for the portfolio
            min_capital_req (float): Minimum capital requirement for the portfolio
            freq (float, optional): Number of trading days per year. Defaults to 252.

        Note:
            The TIPP strategy dynamically adjusts the allocation between risky and safe assets
            to protect against downside risk while maintaining upside potential. The floor
            represents the minimum acceptable portfolio value, and the multiplier determines
            how aggressively the strategy responds to market movements.
        """

        # Validate that all rate parameters have the same shape (1, N)
        assert rr.shape == rf.shape, "All rate parameters must have the same shape"
        assert len(rr.shape) == 1, "Rate parameters must have shape (N,)"

        self._capital = capital
        self._multiplier = multiplier
        self._rr = rr
        self._rf = rf
        self._lock_in = lock_in
        self._min_risk_req = min_risk_req
        self._min_capital_req = min_capital_req
        self._freq = freq

        self._portfolio: Optional[NDArray[np.float64]] = None
        self._ref_capital: Optional[NDArray[np.float64]] = None
        self._margin_trigger: Optional[NDArray[np.float64]] = None
        self._floor: Optional[NDArray[np.float64]] = None

    @property
    def portfolio(self) -> np.ndarray | None:
        return self._portfolio

    @property
    def ref_capital(self) -> np.ndarray | None:
        return self._ref_capital

    @property
    def margin_trigger(self) -> np.ndarray | None:
        return self._margin_trigger

    @property
    def floor(self) -> np.ndarray | None:
        return self._floor

    @property
    def min_risk_req(self) -> float | None:
        return self._min_risk_req

    @property
    def min_capital_req(self) -> float | None:
        return self._min_capital_req

    @property
    def lock_in(self) -> float | None:
        return self._lock_in

    @property
    def rr(self) -> np.ndarray | None:
        return self._rr

    @property
    def rf(self) -> np.ndarray | None:
        return self._rf

    def run(self) -> None:
        """Run the TIPP strategy.

        This method executes the TIPP strategy by dynamically adjusting the portfolio
        allocation between risky and safe assets based on market conditions. It updates
        the floor, lock-in, and portfolio values at each time step.
        """

        compounded_period = self._rr.size / self._freq
        discount = (
            1 + np.float64(self._rf[0]) * self._freq / OPEN_DAYS_PER_YEAR
        ) ** compounded_period

        self._floor = (
            np.ones(self._rr.size) * self._capital * self._min_capital_req / discount
        )
        self._margin_trigger = np.zeros(self._rr.size)
        self._ref_capital = np.ones(self._rr.size) * self._capital
        self._portfolio = np.ones(self._rr.size) * self._capital

        for i in range(1, self._portfolio.size):
            if self._should_update_lock_in(i - 1):
                self._ref_capital[i] = self._portfolio[i - 1]
            else:
                self._ref_capital[i] = self._ref_capital[i - 1]

            floor_cap = self._portfolio[i - 1] * discount

            if self._should_update_floor(floor_cap, i - 1):
                self._floor[i] = floor_cap
            else:
                self._floor[i] = self._floor[i - 1]

            if self._should_inject_liquidity(i - 1):
                capital_to_inject = (
                    self._ref_capital[i - 1] * self._min_capital_req
                    - self._portfolio[i - 1]
                )
                self._ref_capital[i - 1] = self._ref_capital[i - 1] - capital_to_inject
                self._portfolio[i - 1] += capital_to_inject
                self._margin_trigger[i - 1] = capital_to_inject

            magnet = self._portfolio[i - 1] - self._floor[i - 1]
            risk_allocation = self._get_risk_allocation_mix(magnet, i - 1)
            risk_free_allocation = self._portfolio[i - 1] - risk_allocation
            self._portfolio[i] = self._update_portfolio_mix(
                risk_allocation, risk_free_allocation, i
            )
            compounded_period -= 1 / self._freq
            discount = (
                1 + np.float64(self._rf[i]) * self._freq / OPEN_DAYS_PER_YEAR
            ) ** compounded_period

    def _should_update_lock_in(self, n: int) -> bool:
        assert self._portfolio is not None and self._ref_capital is not None
        return self._portfolio[n] >= (1 + self._lock_in) * self._ref_capital[n]

    def _should_update_floor(self, floor_cap: float, n: int) -> bool:
        assert self._floor is not None
        return floor_cap > self._floor[n]

    def _should_inject_liquidity(self, n: int) -> bool:
        assert self._portfolio is not None and self._ref_capital is not None
        return self._portfolio[n] < self._ref_capital[n] * self._min_capital_req

    def _get_risk_allocation_mix(self, magnet: float, n: int) -> float:
        assert self._portfolio is not None
        return max(
            min(self._multiplier * magnet, self._portfolio[n]),
            self._min_risk_req * self._portfolio[n],
        )

    def _update_portfolio_mix(
        self, risk_alloc: float, risk_free_alloc: float, n: int
    ) -> float:
        return risk_alloc * (1 + self._rr[n]) + risk_free_alloc * (1 + self._rf[n])
