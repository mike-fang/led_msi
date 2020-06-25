import numpy as np

def get_intervals(periods, tmax):
    intervals = []
    for n, T in enumerate(periods):
        for t in np.arange(0, tmax, T):
            intervals.append((n, t))
    intervals.sort(key=lambda x:x[1])
    return intervals, get_diff(intervals)

def get_diff(intervals):
    n, t0 = intervals[0]
    diff = [(n, 0)]
    for (n, t) in intervals[1:]:
        dt = t - t0
        diff.append((n, dt))
        t0 = t
    return diff

periods = [.2, .3, .5]
intervals, diff = (get_intervals(periods, 2))
print(diff)
