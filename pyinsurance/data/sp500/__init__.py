from pandas import DataFrame

from pyinsurance.data.utils import load_file


def load() -> DataFrame:
    """
    Load the sp500 data used in the examples
    Returns
    -------
    data : DataFrame
        Data set containing OHLC, adjusted close and the trading volume.
    """
    return load_file(__file__, "sp500.csv.gz")
