import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def _load_rf(_raw_path="../eco_overnight_intermediated_transactions.csv"):
    return pd.read_csv(_raw_path, index_col=0, parse_dates=True)


def _load_future(_raw_path="../future.xlsx"):
    future = pd.read_excel(_raw_path, parse_dates=True, index_col=0, header=12).iloc[1:, :7]
    future.index = pd.to_datetime(future.index)
    future["start_index"] = future["Days to maturity()"].diff() / future["Days to maturity()"].diff().abs()
    return future


def _load_future_start_date():
    future = _load_future()
    future_start_date = future.loc[future["start_index"] == 1].index
    return future_start_date


def _test():
    rf = _load_rf()

    future = _load_future()
    future_start_date = _load_future_start_date()

    hist = future.loc[future_start_date[-6]: future_start_date[-2]][:-1]
    hist_under_prc = hist["Underlying asset price(Pt.)"]
    hist_mu = hist_under_prc.pct_change().mean()
    hist_vol = hist_under_prc.pct_change().std()

    current = future.loc[future_start_date[-2]: future_start_date[-1]][:-1]
    current_close_prc = current["Close Prc.(Pt.)"]

    dt_to_maturity = len(current)
    t = dt_to_maturity / 252

    def simulation(s0):
        s1 = s0 + s0 * (
                hist_mu + (hist_vol * np.random.normal(size=10000))
        )
        return s1
    init = np.array(current_close_prc.iloc[[0]].repeat(10000)).reshape(1,-1).flatten()
    result = []
    s0 = init
    result.append(s0)
    s1 = simulation(s0)
    result.append(s1)
    for i in tqdm(range(2, dt_to_maturity)):
        s1 = simulation(s1)
        result.append(s1)

    final = pd.DataFrame(result, index=current.index)
    final_mean = final.mean(axis=1)

    temp_rf = rf.loc[final_mean.index[0]] / 100
    (final_mean.iloc[-1] * np.exp(-temp_rf * t) / final_mean[0]).values

    plt.plot(final)
    plt.show()





