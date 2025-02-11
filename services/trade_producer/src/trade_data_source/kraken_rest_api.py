from typing import List, Optional, Tuple
from loguru import logger
import json
from time import sleep
from pathlib import Path

import requests

from src.trade_data_source.base import TradeSource, Trade

class KrakenRestAPI(TradeSource):

    URL = 'https://api.kraken.com/0/public/Trades?pair={product_id}&since={since_sec}'

    def __init__(
        self,
        product_id: str,
        last_n_days: int,
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Basic initialization of the Kraken Rest API.

        Args:
            product_id (str): One product ID for which we want to get the trades.
            last_n_days (int): The number of days from which we want to get historical data.
            cache_dir (Optional[str]): The directory where we will store the historical data to

        Returns:
            None
        """
        self.product_id = product_id
        self.from_ms, self.to_ms = self._init_from_to_ms(last_n_days)

        logger.debug(
            f'Initializing KrakenRestAPI: from_ms={ts_to_date(self.from_ms)}, to_ms={ts_to_date(self.to_ms)}'
        )

        self.last_trade_ms = self.from_ms


        self.use_cache = False
        if cache_dir is not None:
            self.cache = CachedTradeData(cache_dir)
            self.use_cache = True

    @staticmethod
    def _init_from_to_ms(last_n_days: int) -> Tuple[int, int]:
        """
        Returns the from_ms and to_ms timestamps for the historical data.
        These values are computed using today's date at midnight and the last_n_days.

        Args:
            last_n_days (int): The number of days from which we want to get historical data.

        Returns:
            Tuple[int, int]: A tuple containing the from_ms and to_ms timestamps.
        """
        # get the current date at midnight using UTC
        from datetime import datetime, timezone

        today_date = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        # today_date to milliseconds
        to_ms = int(today_date.timestamp() * 1000)

        # from_ms is last_n_days ago from today, so
        from_ms = to_ms - last_n_days * 24 * 60 * 60 * 1000

        return from_ms, to_ms

    def get_trades(self) -> List[Trade]:
        """
        Fetches a batch of trades from the Kraken Rest API and returns them as a list
        of dictionaries.

        Args:
            None

        Returns:
            List[Trade]: A list of dictionaries, where each dictionary contains the trade data.
        """
        # Replace the placeholders in the URL with the actual values for
        # - product_id
        # - since_ns
        since_ns = self.last_trade_ms * 1_000_000
        payload = {}
        headers = {'Accept': 'application/json'}
        url = self.URL.format(product_id=self.product_id, since_sec=since_ns)
        logger.debug(f'{url=}')

        if self.use_cache and self.cache.has(url):
            # read the data from the cache
            trades = self.cache.read(url)
            logger.debug(
                f'Loaded {len(trades)} trades for {self.product_id}, since={ns_to_date(since_ns)} from the cache'
            )
        else:
            # make the request to the Kraken REST API
            response = requests.request('GET', url, headers=headers, data=payload)

            # parse string into dictionary
            data = json.loads(response.text)


            # Error Handling 
            if ('error' in data) and ('EGeneral:Too many requests' in data['error']):
                # slow down the rate at which we are making requests to the Kraken API
                logger.info('Too many requests. Sleeping for 30 seconds')
                sleep(30)


            trades = [
                Trade(
                    price=float(trade[0]),
                    quantity=float(trade[1]),
                    timestamp_ms=int(trade[2] * 1000),
                    product_id=self.product_id,
                )
                for trade in data['result'][self.product_id]
            ]

            logger.debug(
                f'Fetched {len(trades)} trades for {self.product_id}, since={ns_to_date(since_ns)} from the Kraken REST API'
            )

            if self.use_cache:
                # write the data to the cache
                self.cache.write(url, trades)
                logger.debug(
                    f'Wrote to cache for {self.product_id}, since={ns_to_date(since_ns)}'
                )

            # slow down the rate at which we are making requests to the Kraken API
            sleep(1)

        if trades[-1].timestamp_ms == self.last_trade_ms:

            self.last_trade_ms = trades[-1].timestamp_ms + 1
        else:

            self.last_trade_ms = trades[-1].timestamp_ms
        
        # filter out trades that are after the end timestamp
        trades = [trade for trade in trades if trade.timestamp_ms <= self.to_ms]


        return trades

    def is_done(self) -> bool:
        # return self._is_done
        return self.last_trade_ms >= self.to_ms


class CachedTradeData:
    """
    A class to handle the caching of trade data fetched from the Kraken REST API.
    """

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = Path(cache_dir)

        if not self.cache_dir.exists():
            # create the cache directory if it does not exist
            self.cache_dir.mkdir(parents=True)

    def read(self, url: str) -> List[Trade]:
        """
        Reads from the cache the trade data for the given url
        """
        file_path = self._get_file_path(url)

        if file_path.exists():
            # read the data from the parquet file
            import pandas as pd

            data = pd.read_parquet(file_path)
            # transform the data to a list of Trade objects
            return [Trade(**trade) for trade in data.to_dict(orient='records')]

        return []

    def write(self, url: str, trades: List[Trade]) -> None:
        """
        Saves the given trades to a parquet file in the cache directory.
        """
        if not trades:
            return

        # transform the trades to a pandas DataFrame
        import pandas as pd

        data = pd.DataFrame([trade.model_dump() for trade in trades])

        # write the DataFrame to a parquet file
        file_path = self._get_file_path(url)
        data.to_parquet(file_path)

    def has(self, url: str) -> bool:
        """
        Returns True if the cache has the trade data for the given url, False otherwise.
        """
        file_path = self._get_file_path(url)
        return file_path.exists()

    def _get_file_path(self, url: str) -> str:
        """
        Returns the file path where the trade data for the given url is (or will be) stored.
        """
        # use the given url to generate a unique file name in a deterministic way
        import hashlib

        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f'{url_hash}.parquet'
        # return self.cache_dir / f'{product_id.replace ("/","-")}_{from_ms}.parquet'


def ts_to_date(ts: int) -> str:
    """
    Transform a timestamp in Unix milliseconds to a human-readable date

    Args:
        ts (int): A timestamp in Unix milliseconds

    Returns:
        str: A human-readable date in the format '%Y-%m-%d %H:%M:%S'
    """
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime(
        '%Y-%m-%d %H:%M:%S'
    )


def ns_to_date(ns: int) -> str:
    """
    Transform a timestamp in Unix nanoseconds to a human-readable date

    Args:
        ns (int): A timestamp in Unix nanoseconds

    Returns:
        str: A human-readable date in the format '%Y-%m-%d %H:%M:%S'
    """
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ns / 1_000_000_000, tz=timezone.utc).strftime(
        '%Y-%m-%d %H:%M:%S'
    )