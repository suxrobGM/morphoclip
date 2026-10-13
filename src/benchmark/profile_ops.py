"""Lightweight profile-frame operations shared across the stable benchmark.

Kept dependency-free (pandas only) so the pure result-accumulation logic can be
imported and unit-tested without the optional ``benchmark`` extra (copairs,
scikit-learn, matplotlib).
"""

import pandas as pd


def concat_profiles(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df2* if *df1* is empty, else inner-join concat both."""
    if df1.shape[0] == 0:
        return df2.copy()
    return pd.concat([df1, df2], ignore_index=True, join="inner")
