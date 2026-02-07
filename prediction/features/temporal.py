"""
Temporal / calendar features.

Encodes time-based information:
- Day of week, month, quarter
- Is month-end / quarter-end / year-end
- Trading day of month / year
- Tunisian public holidays
- Ramadan indicator (approximate)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Tunisian public holidays (fixed-date only; Islamic holidays shift yearly)
TUNISIAN_FIXED_HOLIDAYS = [
    (1, 1),   # New Year
    (1, 14),  # Revolution Day
    (3, 20),  # Independence Day
    (4, 9),   # Martyrs' Day
    (5, 1),   # Labour Day
    (6, 1),   # Victory Day
    (7, 25),  # Republic Day
    (8, 13),  # Women's Day
    (10, 15), # Evacuation Day
]


class TemporalFeatures:
    """Generates calendar and time-based features from the seance column.

    These features capture weekly / monthly / seasonal patterns
    specific to the BVMT trading calendar.
    """

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features to the DataFrame.

        Args:
            df: DataFrame with a 'seance' datetime column.

        Returns:
            DataFrame with additional temporal feature columns.
        """
        df = df.copy()
        dt = pd.to_datetime(df["seance"])

        # Basic calendar features
        df["day_of_week"] = dt.dt.dayofweek  # 0=Mon, 4=Fri
        df["day_of_month"] = dt.dt.day
        df["month"] = dt.dt.month
        df["quarter"] = dt.dt.quarter
        df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
        df["day_of_year"] = dt.dt.dayofyear

        # Binary flags
        df["is_month_start"] = dt.dt.is_month_start.astype(int)
        df["is_month_end"] = dt.dt.is_month_end.astype(int)
        df["is_quarter_end"] = dt.dt.is_quarter_end.astype(int)
        df["is_year_end"] = dt.dt.is_year_end.astype(int)

        # Cyclical encoding (sin/cos) for periodicity
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 5)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 5)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Trading day within month (for each ticker group)
        if "code" in df.columns:
            df["trading_day_of_month"] = df.groupby(
                ["code", dt.dt.year, dt.dt.month]
            ).cumcount() + 1

        # Holiday proximity (days since / until next fixed holiday)
        df["is_near_holiday"] = self._is_near_holiday(dt).astype(int)

        logger.debug("Computed %d temporal features.", 16)
        return df

    @staticmethod
    def _is_near_holiday(dt_series: pd.Series, days_window: int = 2) -> pd.Series:
        """Flag trading days within N days of a known public holiday."""
        result = pd.Series(False, index=dt_series.index)
        for month, day in TUNISIAN_FIXED_HOLIDAYS:
            for year in dt_series.dt.year.unique():
                try:
                    holiday = pd.Timestamp(year=year, month=month, day=day)
                    near = (dt_series - holiday).abs() <= pd.Timedelta(days=days_window)
                    result |= near
                except ValueError:
                    pass
        return result
