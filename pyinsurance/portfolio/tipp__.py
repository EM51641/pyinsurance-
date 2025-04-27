import numpy as np


class TIPP:
    """Time Invariant Portfolio Protection (TIPP) implementation.

    This class implements the core TIPP strategy logic, managing the dynamic
    allocation between risky and safe assets to protect against downside risk
    while maintaining upside potential.

    Attributes:
        params: TIPP model parameters
        tipp: Tuple containing the TIPP model results
    """

    def __init__(
        self,
        capital: float,
        multiplier: float,
        rr: np.float64,
        rf: np.float64,
        br: np.float64,
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
            rr (np.float64): Return rate of the risky asset
            rf (np.float64): Risk-free rate of return
            br (np.float64): Benchmark return rate
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
        assert (
            rr.shape == rf.shape == br.shape
        ), "All rate parameters must have the same shape"
        assert len(rr.shape) == 1, "Rate parameters must have shape (N,)"

        self._capital = capital
        self._multiplier = multiplier
        self._rr = rr
        self._rf = rf
        self._br = br
        self._lock_in = lock_in
        self._min_risk_req = min_risk_req
        self._min_capital_req = min_capital_req
        self._freq = freq
        self._portfolio = self._ref_capital = self._margin_trigger = self._floor = (
            self._discount
        ) = self._compounded_period = None

    @property
    def portfolio(self):
        return self._portfolio

    @property
    def ref_capital(self):
        return self._ref_capital

    @property
    def margin_trigger(self):
        return self._margin_trigger

    @property
    def floor(self):
        return self._floor

    @property
    def compounded_period(self):
        return self._compounded_period

    @property
    def discount(self):
        return self._discount

    @property
    def min_risk_req(self):
        return self._min_risk_req

    @property
    def min_capital_req(self):
        return self._min_capital_req

    @property
    def lock_in(self):
        return self._lock_in

    @property
    def rr(self):
        return self._rr

    @property
    def rf(self):
        return self._rf

    @property
    def br(self):
        return self._br

    def run(self):
        """Run the TIPP strategy.

        This method executes the TIPP strategy by dynamically adjusting the portfolio
        allocation between risky and safe assets based on market conditions. It updates
        the floor, lock-in, and portfolio values at each time step.
        """
        self._compounded_period = self._rr.size / self._freq
        self._discount = (1 + self._rf * self._freq / 252) ** self._compounded_period
        self._floor = self._capital * self._min_capital_req / self._discount
        self._margin_trigger = np.zeros(self._rr.size)
        self._ref_capital = np.ones(self._rr.size) * self._capital
        self._portfolio = np.ones(self._rr.size) * self._capital

        for i in range(1, self._portfolio.size):
            if self._should_update_lock_in(i - 1):
                self._ref_capital[i] = self._portfolio[i - 1]
            else:
                self._ref_capital[i] = self._ref_capital[i - 1]

            floor_cap = self._portfolio[i - 1] * self._discount[i - 1]

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
            self._compounded_period -= 1 / self._freq

    def _should_update_lock_in(self, n):
        return self._portfolio[n] >= (1 + self._lock_in) * self._ref_capital[n]

    def _should_update_floor(self, floor_cap, n):
        return floor_cap > self._floor[n]

    def _should_inject_liquidity(self, n):
        return self._portfolio[n] < self._ref_capital[n] * self._min_capital_req

    def _get_risk_allocation_mix(self, magnet, n):
        return max(
            min(self._multiplier * magnet, self._portfolio[n]),
            self._min_risk_req * self._portfolio[n],
        )

    def _update_portfolio_mix(self, risk_alloc, risk_free_alloc, n):
        return risk_alloc * (1 + self._rr[n]) + risk_free_alloc * (1 + self._rf[n])
