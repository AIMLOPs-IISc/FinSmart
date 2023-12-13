import os
import random
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from datetime import datetime, timedelta
import pickle

class Predictor:
    def __init__(self):
        self.data_path = os.path.join(os.path.dirname(__file__), "data")
        self.model_path = os.path.join(os.path.dirname(__file__), "xgb_model.pkl")
        self.model = pickle.load(open(self.model_path, "rb"))
        self.ndays = 20
        self.train_test_split_ratio = 0.8
        self.train_files =[]
        self.test_files =[]
        self.train_test_split()
        self.n_estimators = 250
        self.max_depth = 5
        self.eta = 0.1
        self.subsample = 0.7
        self.colsample_bytree = 0.8
        self.objective = 'reg:squarederror'
        self.random_state = 11

    def train_test_split(self):
        files = []
        for fname in os.listdir(self.data_path):
            fpath = os.path.join(self.data_path, fname)
            if os.path.isfile(fpath):
                if fpath.endswith("_price.csv"):
                    files.append(fpath)
        random.shuffle(files)
        cnt = int(len(files) * self.train_test_split_ratio)
        self.train_files = files[:cnt]
        self.test_files = files[cnt:]

    def train(self):
        trg = []
        teg = []
        start = datetime.now()
        for file in self.train_files:
            df = pd.read_csv(file)
            X, y = self.get_shifted(df, self.ndays, True)
            trg.append(X)
            teg.append(y)
        train_X = pd.concat(trg)
        train_y = pd.concat(teg)

        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            eta=self.eta,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective=self.objective,
            random_state=self.random_state
        )
        start = datetime.now()
        self.model.fit(train_X, train_y)
        yb = self.model.predict(train_X)
        pred = pd.DataFrame(yb, columns=["Open", "Close", "High", "Low", "Volume"])
        run_time = datetime.now() - start
        print(r2_score(train_y, pred), "Runtime:", run_time)

        # save
        pickle.dump(self.model, open(self.model_path, "wb"))

    def evaluate(self):
        sut = random.choice(self.test_files)
        df = pd.read_csv(sut)
        X, y = self.get_shifted(df, self.ndays, True)
        if len(X) > 0:
            yb = self.model.predict(X)
            pred = pd.DataFrame(yb, columns=["Open", "Close", "High", "Low", "Volume"])
            return r2_score(y, pred)
        return 1


    def increase_pred(self, Curr, pred):
        da = list(Curr.iloc[-1][["Year", "Month", "Day"]].astype(int).astype(str))
        dt = datetime.strptime("-".join(da), "%Y-%m-%d")
        dt = dt + timedelta(days=1)
        if dt.weekday() == 5:
            dt = dt + timedelta(days=2)
        elif dt.weekday() == 6:
            dt = dt + timedelta(days=1)
        ostr = dt.strftime("%Y-%m-%d")
        new_list = list(pred.iloc[-1]) \
                   + list(Curr.iloc[-1])[3:]
        new_list = new_list[:100]
        final = [int(v) for v in ostr.split("-")] + new_list
        data = pd.DataFrame(final).T
        data.columns = list(Curr.columns)
        return data

    def get_shifted(self, data, day, train=False):
        df = data.copy()
        df[["Year", "Month", "Day"]] = df.Date.str.split("-", expand=True).astype(int)
        add = 1
        Target = ["Open", "Close", "High", "Low", "Volume"]
        df = df[["Year", "Month", "Day"] + Target]
        if not train:
            df = df[-1 * day:]
            add = 0
        dfgroups = []
        for itr in range(add, day + add):
            dfgroups.append(pd.DataFrame())
            dfgroups[itr - add] = df[Target].shift(periods=itr)
            dfgroups[itr - add].columns = [f"{tgt}_{itr - add}" for tgt in Target]
        df = pd.concat([df] + dfgroups, axis=1)
        cols = [f for f in list(df.columns) if f not in Target]
        if train:
            dfn = df[day:].dropna(subset=Target).reset_index(drop=True)
            return dfn[cols], dfn[Target]
        return df.iloc[[-1]][cols].reset_index(drop=True)

    def next_n_pred(self, data, days=10):
        X, y = self.get_shifted(data.sort_values(by="Date"), 20, train=True)
        train = X.copy()
        yb = self.model.predict(train)
        pred = pd.DataFrame(yb, columns=["Open", "Close", "High", "Low", "Volume"])
        predc = pred.copy()
        score = r2_score(y, pred)
        new_x = self.increase_pred(X, pred)
        for i in range(days):
            train = pd.concat([train, new_x])
            yb_n = self.model.predict(new_x)
            pred = pd.DataFrame(yb_n, columns=["Open", "Close", "High", "Low", "Volume"])
            new_x = self.increase_pred(train, pred)
            predc = pd.concat([predc, pred])
        y.columns = [f"True_{val}" for val in list(y.columns)]
        predc.columns = [f"Pred_{val}" for val in list(predc.columns)]
        datecols = train[["Year", "Month", "Day"]].astype(int).astype(str)
        datecols["Date"] = datecols["Year"] + "-" + datecols["Month"] + "-" + datecols["Day"]
        final = pd.concat([datecols[["Date"]], y, predc], axis=1)
        return final.reset_index(drop=True), score