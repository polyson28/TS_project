import math
from collections import deque


class UniversalChangePointDetector:
    """
    Universal change-point detector for non‑i.i.d. data with target false‑alarm ≤10%.
    Combines triple exponential smoothing (ES), CUSUM (mean & variance), Shiryaev–Roberts.
    """

    def __init__(self,
                 threshold=2.0, 
                 direction="both",
                 season_length=24,
                 alpha=0.2, beta=0.1, gamma=0.05,
                 k_mean=0.5, k_var=0.5,
                 max_buffer=200,
                 warmup=50):
        # the initial statistics threshold, will be calibrated in self.calibrate method
        self.threshold = threshold
        self.direction = direction  # 'increase', 'decrease', or 'both', which of the directions is needed to be controlled

        # Triple‑ES params
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.season_length = season_length

        # CUSUM reference drifts
        self.k_mean = k_mean
        self.k_var = k_var

        # State variables
        self.level = None # current smoothing level
        self.trend = 0.0  # current trend
        self.seasonals = [1.0] * season_length
        self.residuals = deque(maxlen=max_buffer)

        # Statistics
        self.cusum_up = 0.0
        self.cusum_down = 0.0
        self.cusum_var_up = 0.0
        self.cusum_var_down = 0.0
        self.sr = 1.0

        # Warm‑up - The number of the first points for which we only "warm up" the model without issuing signals
        self.count = 0
        self.warmup = warmup

    @classmethod
    def calibrate(cls, historical_series, **kwargs):
        """
        Calibrate detection threshold on historical data to fix
        false‑alarm rate at ≈10%. Returns a configured detector.
        """
        det = cls(**kwargs)
        scores = []
        for t, x in enumerate(historical_series):
            score = det.update(x, calibrate=True)
            if t >= det.warmup:
                scores.append(score)
        # Set threshold at 90th percentile of in‑control scores
        det.threshold = sorted(scores)[int(0.9 * len(scores))]
        # Reset before actual monitoring
        det.reset_state()
        return det

    def reset_state(self):
        """Reset all streaming state except threshold & config."""
        self.level = None
        self.trend = 0.0
        self.seasonals = [1.0] * self.season_length
        self.residuals.clear()
        self.cusum_up = self.cusum_down = 0.0
        self.cusum_var_up = self.cusum_var_down = 0.0
        self.sr = 1.0
        self.count = 0

    def update(self, value, calibrate=False):
        """
        Process a new observation.
        Returns the current alarm score (stat/threshold).
        """
        season_idx = self.count % self.season_length
        if self.level is None:
            # Initialize on first point
            self.level = value
            self.residuals.append(0.0)
            self.count += 1
            return 0.0

        # Forecast + residual
        forecast = (self.level + self.trend) * self.seasonals[season_idx]
        resid = value - forecast
        self.residuals.append(resid)

        # Triple‑ES update
        prev_level = self.level
        self.level = (self.alpha * (value / self.seasonals[season_idx]) +
                      (1 - self.alpha) * (prev_level + self.trend))
        self.trend = (self.beta * (self.level - prev_level) +
                      (1 - self.beta) * self.trend)
        self.seasonals[season_idx] = (
            self.gamma * (value / self.level) +
            (1 - self.gamma) * self.seasonals[season_idx]
        )

        # Estimate mean & std from buffer
        m = sum(self.residuals) / len(self.residuals)
        ss = sum((r - m)**2 for r in self.residuals) / len(self.residuals)
        sigma = math.sqrt(ss) if ss > 1e-6 else 1.0
        z = (resid - m) / sigma

        # Update mean‑CUSUM
        if self.direction in ("increase", "both"):
            self.cusum_up = max(0.0, self.cusum_up + z - self.k_mean)
        if self.direction in ("decrease", "both"):
            self.cusum_down = max(0.0, self.cusum_down - z - self.k_mean)

        # Update var‑CUSUM
        vstat = z*z - 1.0
        self.cusum_var_up = max(0.0, self.cusum_var_up + vstat - self.k_var)
        self.cusum_var_down = max(0.0, self.cusum_var_down - vstat - self.k_var)

        # Update Shiryaev–Roberts
        lr = z - self.k_mean/2
        if lr > -20:
            self.sr = (1 + self.sr) * math.exp(lr)
        else:
            self.sr = 0.0

        # Aggregate
        if self.direction == "increase":
            cus = max(self.cusum_up, self.cusum_var_up)
        elif self.direction == "decrease":
            cus = max(self.cusum_down, self.cusum_var_down)
        else:
            cus = max(self.cusum_up, self.cusum_down,
                      self.cusum_var_up, self.cusum_var_down)

        # weighted score
        stat = 0.5 * cus + 0.5 * self.sr
        alarm = stat / self.threshold

        self.count += 1
        return alarm if calibrate else (alarm > 1.0)

    @property
    def debug(self):
        return {
            "level": self.level,
            "trend": self.trend,
            "cusum_up": self.cusum_up,
            "cusum_down": self.cusum_down,
            "cusum_var_up": self.cusum_var_up,
            "cusum_var_down": self.cusum_var_down,
            "sr": self.sr,
            "threshold": self.threshold
        }
