import numpy as np

def bilin_2d(data: np.ndarray, interval: int, lon_mode=False):
    data = data.copy()

    if interval == 1:
        return data

    if lon_mode is True:
        max_diff = np.nanmax(np.abs(data[:, :-1] - data[:,1:]))
        if max_diff > 180.:
            data[data < 0] = 360. + data[data < 0]

    data = np.concatenate((data, data[-1].reshape(1, -1)), axis=0)
    data = np.concatenate((data, data[:, -1].reshape(-1, 1)), axis=1)

    ratio_horizontal = np.tile(np.linspace(0, (interval - 1) / interval, interval, dtype=np.float32),
                               (data.shape[0] * interval, data.shape[1] - 1))
    ratio_vertical = np.tile(np.linspace(0, (interval - 1) / interval, interval, dtype=np.float32).reshape(-1, 1),
                             (data.shape[0] - 1, (data.shape[1] - 1) * interval))
    repeat_data = np.repeat(data, interval, axis=0)
    repeat_data = np.repeat(repeat_data, interval, axis=1)

    horizontal_interp = (1. - ratio_horizontal) * repeat_data[:, :-interval] + ratio_horizontal * repeat_data[:, interval:]
    ret = (1. - ratio_vertical) * horizontal_interp[:-interval, :] + ratio_vertical * horizontal_interp[interval:, :]

    if lon_mode is True:
        ret[ret > 180.] = ret[ret > 180.] - 360.

    return ret

# EOF
