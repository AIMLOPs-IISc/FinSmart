import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import shutil
from requests.exceptions import ConnectTimeout, JSONDecodeError

class NSE:
    def __init__(self):
        self.__local_path__ = os.path.dirname(__file__)
        self.__data_path__ = os.path.join(self.__local_path__, "data")
        self.stocks = pd.read_csv(os.path.join(self.__data_path__, "company_list.csv"))
        self.__base_url__ = "https://www.nseindia.com"
        self.__headers__ = {
            'Connection': 'keep-alive',
            'Cache-Control': 'max-age=0',
            'DNT': '1',
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0',
            'Sec-Fetch-User': '?1',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-Mode': 'navigate',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9,hi;q=0.8',
            'x-requested-with': 'XMLHttpRequest',
            'referer': 'https://www.nseindia.com/get-quotes/equity?symbol=COALINDIA'
        }
        self.__session__ = requests.session()
        #set cookies
        self.nsefetch("")

    @property
    def datapath(self):
        return self.__data_path__

    @datapath.setter
    def datapath(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        for fname in os.listdir(self.__data_path__):
            src = os.path.join(self.__data_path__, fname)
            dest = os.path.join(path, fname)
            shutil.copyfile(src, dest)
        self.__data_path__ = path
        self.stocks = pd.read_csv(os.path.join(self.__data_path__, "company_list.csv"))

    def tickers(self):
        return list(self.stocks["Symbol"].unique())

    def company(self, symbol):
        ndf = self.stocks[self.stocks.Symbol.str.lower() == symbol.lower()].copy()
        if len(ndf) == 1:
            return ndf["Company Name"].values[0]
        return "na"

    def nsefetch(self, url):
        print("Calling NSE", self.__base_url__ + url)
        try:
            data = self.__session__.get(self.__base_url__ + url, headers=self.__headers__, timeout=5)
            resp = data.json()
            df = pd.DataFrame.from_records(resp["data"])
            print("Success")
        except ConnectTimeout:
            df = pd.DataFrame()
            print("Fail - Timeout")
        except JSONDecodeError:
            df = pd.DataFrame()
            print("Fail - JSON Decode")
        return df


    def local_data(self, symbol):
        dpath = os.path.join(self.__data_path__, f"{symbol}_price.csv")
        if os.path.exists(dpath):
            return dpath, pd.read_csv(dpath)
        return dpath, pd.DataFrame()


    @staticmethod
    def update_local(dpath, df):
        curr_df = pd.DataFrame
        if df != pd.DataFrame():
            if os.path.exists(dpath):
                curr_df = pd.read_csv(dpath)
                curr_df = curr_df.append(df, ignore_index=True, )
                curr_df = curr_df.drop_duplicates(subset =["CH_TIMESTAMP"], keep="last")
            else:
                curr_df = df
            curr_df = curr_df.sort_values(by="CH_TIMESTAMP", ascending=False)
            curr_df.reset_index(drop=True, inplace=True)
            curr_df.to_csv(dpath, index=False)
        return curr_df


    def historical(self, symbol, days, series="EQ"):
        dpath, df = self.local_data(symbol)
        online = False
        et = datetime.now()
        end_date = et.strftime("%Y-%m-%d")
        start_date = (et - timedelta(days=days)).strftime("%Y-%m-%d")
        if df == pd.DataFrame():
            online = True
        else:
            online = ~((start_date in df["CH_TIMESTAMP"]) and (end_date in df["CH_TIMESTAMP"]))
        if online:
            mnths = int(days / 30)
            days = days % 30
            batch = [30] * mnths + [days]
            current_end_date = et
            start_end_batch = []
            for day_count in batch:
                end_date_t = current_end_date
                start_date_t = end_date_t - timedelta(days=day_count)
                current_end_date = start_date_t
                start_end_batch.append((start_date_t.strftime("%d-%m-%Y"),end_date_t.strftime("%d-%m-%Y")))
            for fetch_batch in start_end_batch:
                url = '/api/historical/cm/equity?symbol=' + symbol + '&series=["' + series + '"]&from=' + fetch_batch[0] + '&to=' + fetch_batch[1]
                df = self.nsefetch(url)
                columns = [
                    "CH_TIMESTAMP",
                    "CH_SYMBOL",
                    "CH_SERIES",
                    "CH_TRADE_HIGH_PRICE",
                    "CH_TRADE_LOW_PRICE",
                    "CH_OPENING_PRICE",
                    "CH_CLOSING_PRICE",
                    "CH_TOT_TRADED_QTY",
                    "CH_TOT_TRADED_VAL"
                ]
                df = self.update_local(dpath, df[columns])
        if df == pd.DataFrame():
            return df
        mask = (df["CH_TIMESTAMP"] > start_date) & (df["CH_TIMESTAMP"] <= end_date)
        return df.loc[mask]


    def highlights(self, count=5):
        df = self.nsefetch('/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O')
        try:
            df = df[["symbol", "open", "dayHigh", "dayLow", "lastPrice", "totalTradedVolume", "totalTradedValue", "pChange", 'yearHigh', 'yearLow']]
            loosers = df.sort_values(by="pChange").head(count)
            gainers = df.sort_values(by="pChange", ascending=False).head(count)
            active_val = df.sort_values(by="totalTradedValue", ascending=False).head(count)
            active_vol = df.sort_values(by="totalTradedVolume", ascending=False).head(count)
            return {"top_loosers":loosers, "top_gainers":gainers, "most_active_by_value":active_val, "most_active_by_volume":active_vol}
        except:
            return {}


