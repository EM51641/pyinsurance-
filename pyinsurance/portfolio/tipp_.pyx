# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
import numpy as np
cimport numpy as np
from libc.math cimport fmax, fmin

ctypedef np.float64_t DTYPE_t

cdef class TIPP:
    """Time Invariant Portfolio Protection (TIPP) implementation.

    This class implements a dynamic portfolio protection strategy that adjusts the allocation
    between risky and safe assets to protect against downside risk while maintaining upside potential.
    """

    cdef:
        DTYPE_t _capital
        DTYPE_t _multiplier
        np.ndarray _rr
        np.ndarray _rf
        np.ndarray _br
        DTYPE_t _lock_in
        DTYPE_t _min_risk_req
        DTYPE_t _min_capital_req
        DTYPE_t _freq
        np.ndarray _portfolio
        np.ndarray _ref_capital
        np.ndarray _margin_trigger
        np.ndarray _discount
        np.ndarray _floor
        DTYPE_t _compounded_period

    def __init__(
        self,
        DTYPE_t capital,
        DTYPE_t multiplier,
        np.ndarray[DTYPE_t, ndim=1] rr,
        np.ndarray[DTYPE_t, ndim=1] rf,
        np.ndarray[DTYPE_t, ndim=1] br,
        DTYPE_t lock_in,
        DTYPE_t min_risk_req,
        DTYPE_t min_capital_req,
        DTYPE_t freq=252
    ):
        """Initialize TIPP model with parameters."""

    #    # Validate that all rate parameters have the same length
    #    cdef Py_ssize_t rr_len = rr.shape[0]
    #    cdef Py_ssize_t rf_len = rf.shape[0]
    #    cdef Py_ssize_t br_len = br.shape[0]
    #    
    #    if rr_len != rf_len or rr_len != br_len:
    #        raise ValueError("All rate parameters must have the same length")

        self._capital = capital
        self._multiplier = multiplier
        self._rr = rr
        self._rf = rf
        self._br = br
        self._lock_in = lock_in
        self._min_risk_req = min_risk_req
        self._min_capital_req = min_capital_req
        self._freq = freq
        self._portfolio = None
        self._ref_capital = None
        self._margin_trigger = None
        self._discount = None
        self._floor = None
        self._compounded_period = self._rr.size / self._freq

    @property
    def portfolio(self):
        """Get the portfolio value array."""
        return self._portfolio

    @property
    def ref_capital(self):
        """Get the reference capital array."""
        return self._ref_capital

    @property
    def margin_trigger(self):
        """Get the margin trigger array."""
        return self._margin_trigger

    @property
    def floor(self):
        """Get the floor array."""
        return self._floor

    @property
    def compounded_period(self):
        """Get the compounded period."""
        return self._compounded_period

    @property
    def discount(self):
        """Get the discount array."""
        return self._discount

    def run(self):
        """Run the TIPP strategy simulation."""
        cdef:
            Py_ssize_t i
            DTYPE_t floor_cap, magnet, risk_allocation, risk_free_allocation

        self._discount = (1 + self._rf * self._freq / 252) ** self._compounded_period
        self._margin_trigger = np.zeros(self._rr.size, dtype=np.float64)
        self._ref_capital = np.ones(self._rr.size, dtype=np.float64) * self._capital
        self._portfolio = np.ones(self._rr.size, dtype=np.float64) * self._capital
        self._floor = self._capital * self._min_capital_req / self._discount

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
                    self._ref_capital[i - 1] * self._min_capital_req - self._portfolio[i - 1]
                )
                self._ref_capital[i] = self._ref_capital[i - 1] - capital_to_inject
                self._portfolio[i - 1] += capital_to_inject
                self._margin_trigger[i] = capital_to_inject

            magnet = self._portfolio[i - 1] - self._floor[i - 1]
            risk_allocation = self._get_risk_allocation_mix(magnet, i - 1)
            risk_free_allocation = self._portfolio[i - 1] - risk_allocation
            self._portfolio[i] = self._update_portfolio_mix(
                risk_allocation, risk_free_allocation, i
            )
            self._compounded_period -= 1 / self._freq

    cdef bint _should_update_lock_in(self, Py_ssize_t n):
        """Check if reference capital should be updated based on lock-in condition."""
        return self._portfolio[n] >= (1 + self._lock_in) * self._ref_capital[n]

    cdef bint _should_update_floor(self, DTYPE_t floor_cap, Py_ssize_t n):
        """Check if protection floor should be updated."""
        return floor_cap > self._floor[n]

    cdef bint _should_inject_liquidity(self, Py_ssize_t n):
        """Check if liquidity injection is needed."""
        return self._portfolio[n] < self._ref_capital[n] * self._min_capital_req

    cdef DTYPE_t _get_risk_allocation_mix(self, DTYPE_t magnet, Py_ssize_t n):
        """Calculate the risk allocation mix for the portfolio."""
        return fmax(
            fmin(self._multiplier * magnet, self._portfolio[n]),
            self._min_risk_req * self._portfolio[n]
        )

    cdef DTYPE_t _update_portfolio_mix(self, DTYPE_t risk_alloc, 
                                     DTYPE_t risk_free_alloc, 
                                     Py_ssize_t n):
        """Update portfolio value based on risk and risk-free allocations."""
        return risk_alloc * (1 + self._rr[n]) + risk_free_alloc * (1 + self._rf[n]) 