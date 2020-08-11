import time
import pandas as pd
import numpy as np
from datetime import datetime


def string2timestamp(strings, T=48):
    """
    time_str --> timestamp
    example:
    str = [b'2013070101', b'2013070102'] -->
    [Timestamp('2013-07-01 00:00:00'), Timestamp('2013-07-01 00:30:00')]
    """
    timestamps = []
    time_per_slot = 24.0 / T
    num_per_T = T // 24
    for t in strings:
        year, month, day, slot = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:])-1
        timestamps.append(pd.Timestamp(datetime(year, month, day, hour=int(slot * time_per_slot),
                                                minute=(slot % num_per_T) * int(60.0 * time_per_slot))))
    return timestamps


def timestamp2vec(timestamps):
    """
    timestamp --> vector
    example:
    [b'2018120505', b'2018120106']
    [[0 0 1 0 0 0 0 1],  Wednesday & weekday
    [0 0 0 0 0 1 0 0]]  Saturday & weekend
    """
    # tm_wday range [0, 6], Monday is 0
    vec = [time.strptime(str(t[:8], encoding='utf-8'), '%Y%m%d').tm_wday for t in timestamps]  # python3
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ret.append(v)
    return np.asarray(ret)