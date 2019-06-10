import datetime
from spot.config import TZ_UTC0, deltaLST, deltaLST_date0, ORBIT_TABLE
import math
import numpy as np
from scipy import interpolate

# ========================
# Constant
# ========================
# RSP_EPOCH = datetime.datetime(year=2017, month=12, day=11, hour=22, minute=25, second=34, tzinfo=TZ_UTC0)
RSP_EPOCH = datetime.datetime(year=2018, month=1, day=16, hour=7, minute=57, second=14, tzinfo=TZ_UTC0)

REPEAT_CYCLE = datetime.timedelta(days=34)
NUM_PATH = 485.
PATH_PER_DAY = 14. + (9. / 34.)
TIME_PER_PATH = REPEAT_CYCLE / NUM_PATH
PATH0 = 196
LON0 = 0.292

# ========================
# Reference System for Planning (RSP)
# ========================
def find_RSP_fcst(lat: float, lon: float, year=None, month=None, day=None, orbit_direction: str='D'):
    """
        Find the forecast of Reference System for Planning(RSP) path and scene
    :param lat:
    :param lon:
    :param year:
    :param month:
    :param day:
    :param orbit_direction: 'D' => descending  or 'A' => ascending or 'B' => both
    :return:
    """

    # Parse arguments
    tgt_data = datetime.datetime(year=2018, month=1, day=16, tzinfo=TZ_UTC0)
    # tgt_data = datetime.datetime(year=2019, month=2, day=11, tzinfo=TZ_UTC0)

    rsp = _find_RSP_fcs(lat, lon, tgt_data, orbit_direction)

    print(tgt_data, rsp)
    return None

def _find_RSP_fcs(lat: float, lon: float, tgt_date:datetime.datetime, orbit_direction: str):
    (rsp_paths, path_times) = _get_RSP_paths_on_day(tgt_date.date(), pre_path=True, post_path=True)
    # tgt_date_am0 = tgt_date.replace(hour=0, minute=0, second=0, tzinfo=TZ_UTC0)
    # deltaLST_idx = (tgt_date_am0.date() - deltaLST_date0).days
    # num_paths = int((tgt_date_am0 - RSP_EPOCH) / TIME_PER_PATH)
    # path_time_start = num_paths * TIME_PER_PATH + RSP_EPOCH + datetime.timedelta(seconds=deltaLST[deltaLST_idx])
    #
    # rsp_paths = (((PATH0 + np.arange(num_paths, num_paths + PATH_PER_DAY) * REPEAT_CYCLE.days - 1) % NUM_PATH) + 1).astype(np.int32)
    # path_times = (np.arange(0, rsp_paths.shape[0]) * TIME_PER_PATH) + path_time_start


    print('Path', ':', 'Start time')
    for p, t in zip(rsp_paths, path_times):
        print(p,':', t)

    # orbit data
    # with open('spot/table/GCOMC_1path35.bin', 'r') as f:
    #     sl = 404;
    #     pl = 11;
    #     buf = np.fromfile(f, dtype=np.float32).reshape(2, pl*sl)
    #
    #
    # lon = buf[0].reshape(sl, pl)
    # lat = buf[1].reshape(sl, pl)
    #
    # s1 = round(sl / 4);
    # s2 = round(sl * 3 / 4);
    #
    # print(s1)
    # print(s2)
    #
    # if orbit_direction == 'A':
    #     lin = [np.arange(s2 + 1,sl+1), np.arange(1, s1)+1]
    # else:
    #     lin = (s1 + 1:s2)
    #
    # return (0,0)

def _get_RSP_paths_on_day(tgt_day: datetime.date, pre_path=False, post_path=False):
    """
    :param tgt_day: from Jun. 18, 2018
    :return:
    """
    tgt_date_am0 = datetime.datetime(year=tgt_day.year, month=tgt_day.month, day=tgt_day.day, hour=0, minute=0, second=0, tzinfo=TZ_UTC0)
    deltaLST_idx = (tgt_date_am0.date() - deltaLST_date0).days
    if deltaLST_idx < 0:
        deltaLST_idx = 0
    num_total_paths = int((tgt_date_am0 - RSP_EPOCH) / TIME_PER_PATH) - 1
    path_time_start = num_total_paths * TIME_PER_PATH + RSP_EPOCH + datetime.timedelta(seconds=deltaLST[deltaLST_idx])

    rsp_paths = (((PATH0 + np.arange(num_total_paths, num_total_paths + PATH_PER_DAY + 3) * REPEAT_CYCLE.days - 1) % NUM_PATH) + 1).astype(np.int32)
    path_times = (np.arange(0, rsp_paths.shape[0]) * TIME_PER_PATH) + path_time_start

    today_idx = np.arange(0, rsp_paths.shape[0])
    today_idx = today_idx[np.bitwise_and(path_times >= tgt_date_am0, path_times < (tgt_date_am0 + datetime.timedelta(days=1)))]
    ret_range = [today_idx[0], today_idx[-1]]
    ret_range[0]: ret_range[1] + 1
    if pre_path is True:
        ret_range[0] = ret_range[0] - 1
    if post_path is True:
        ret_range[1] = ret_range[1] + 1

    return (rsp_paths[ret_range[0]: ret_range[1] + 1], path_times[ret_range[0]: ret_range[1] + 1])

def _get_RSP_lat():
    times = ORBIT_TABLE[:, 0]
    lat = ORBIT_TABLE[:, 3]

    return lat, times

def _get_RSP_obs_edge_lat():
    times = ORBIT_TABLE[:, 0]

    return ORBIT_TABLE[:, 2], ORBIT_TABLE[:, 4], times

def _get_RSP_lon_form_paths(paths:np.ndarray):
    times = ORBIT_TABLE[:, 0]
    rel_lon = ORBIT_TABLE[:, 6]

    print(TIME_PER_PATH.total_seconds())
    num_paths = paths.shape[0]
    lons = np.tile(rel_lon.reshape(1, -1), (num_paths,1))
    lons0 = np.tile((LON0 + 360. * paths / NUM_PATH).reshape(-1, 1), (1, lons.shape[1])) * -1
    lons = np.mod(lons + lons0, 360)
    lons[lons > 180] = -360. + lons[lons > 180]

    return lons, times

def _get_RSP_obs_edge_lon_form_paths(paths:np.ndarray):
    times = ORBIT_TABLE[:, 0]
    rel_lon = ORBIT_TABLE[:, 6].reshape(1, -1)
    rel_lon_left = ORBIT_TABLE[:, 5].reshape(1, -1)
    rel_lon_right = ORBIT_TABLE[:, 7].reshape(1, -1)

    num_paths = paths.shape[0]
    lons = np.tile(rel_lon, (num_paths, 1))
    lons_right = lons + rel_lon_right
    lons_left = lons + rel_lon_left
    lons0 = np.tile((LON0 + 360. * paths / NUM_PATH).reshape(-1, 1), (1, lons.shape[1])) * -1

    lons_right = np.mod(lons_right + lons0, 360)
    lons_right[lons_right > 180] = -360. + lons_right[lons_right > 180]

    lons_left = np.mod(lons_left + lons0, 360)
    lons_left[lons_left > 180] = -360. + lons_left[lons_left > 180]

    return lons_left, lons_right, times
