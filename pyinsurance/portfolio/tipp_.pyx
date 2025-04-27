# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False

import numpy as np
cimport numpy as np
from libc.math cimport fmax, fmin, pow

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

        # Validate that all rate parameters have the same length
        cdef Py_ssize_t rr_len = rr.shape[0]
        cdef Py_ssize_t rf_len = rf.shape[0]
        cdef Py_ssize_t br_len = br.shape[0]

        if rr_len != rf_len or rr_len != br_len:
            raise ValueError("All rate parameters must have the same length")

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

    def run(self):
        """Run the TIPP strategy simulation."""
        cdef:
            Py_ssize_t i, n
            DTYPE_t floor_cap, magnet, risk_allocation, risk_free_allocation
            DTYPE_t[::1] rr_view = self._rr
            DTYPE_t[::1] rf_view = self._rf
            DTYPE_t[::1] portfolio_view
            DTYPE_t[::1] ref_capital_view
            DTYPE_t[::1] margin_trigger_view
            DTYPE_t[::1] floor_view
            DTYPE_t lock_in
            DTYPE_t min_capital_req
            DTYPE_t freq
            DTYPE_t compounded_period

        n = self._rr.shape[0]

        # Get memoryviews for efficient access
        portfolio_view = np.ones(n, dtype=np.float64) * self._capital
        ref_capital_view = np.ones(n, dtype=np.float64) * self._capital
        margin_trigger_view = np.zeros(n, dtype=np.float64)
        floor_view = np.ones(n, dtype=np.float64) * (self._capital * self._min_capital_req)
        compounded_period = n / self._freq
        freq = self._freq
        min_capital_req = self._min_capital_req
        multiplier = self._multiplier
        lock_in = self._lock_in
        min_risk_req = self._min_risk_req

        # Calculate discount factor once
        discount_factor = pow(1 + rf_view[0] * freq / 252, compounded_period)
        
        for i in range(1, n):
            # Update reference capital
            if portfolio_view[i-1] >= (1 + lock_in) * ref_capital_view[i-1]:
                ref_capital_view[i] = portfolio_view[i-1]
            else:
                ref_capital_view[i] = ref_capital_view[i-1]

            # Update floor
            floor_cap = portfolio_view[i-1] * discount_factor
            if floor_cap > floor_view[i-1]:
                floor_view[i] = floor_cap
            else:
                floor_view[i] = floor_view[i-1]

            # Check for liquidity injection
            if portfolio_view[i-1] < ref_capital_view[i-1] * min_capital_req:
                capital_to_inject = ref_capital_view[i-1] * min_capital_req - portfolio_view[i-1]
                ref_capital_view[i] = ref_capital_view[i-1] - capital_to_inject
                portfolio_view[i-1] += capital_to_inject
                margin_trigger_view[i] = capital_to_inject

            # Calculate allocations
            magnet = portfolio_view[i-1] - floor_view[i-1]
            risk_allocation = fmax(
                fmin(multiplier * magnet, portfolio_view[i-1]),
                min_risk_req * portfolio_view[i-1]
            )
            risk_free_allocation = portfolio_view[i-1] - risk_allocation
            
            # Update portfolio
            portfolio_view[i] = risk_allocation * (1 + rr_view[i]) + risk_free_allocation * (1 + rf_view[i])
    
            compounded_period -= 1 / freq
            discount_factor = pow(1 + rf_view[i] * freq / 252, compounded_period)

        self._portfolio = np.asarray(portfolio_view)
        self._ref_capital = np.asarray(ref_capital_view)
        self._margin_trigger = np.asarray(margin_trigger_view)
        self._floor = np.asarray(floor_view)
