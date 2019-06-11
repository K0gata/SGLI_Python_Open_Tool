import numpy as np
import math
import enum
import logging
from abc import ABC, abstractmethod, abstractproperty
from decimal import Decimal, ROUND_HALF_UP

class PROJ_TYPE(enum.Enum):
    SCENE = enum.auto()
    TILE = enum.auto()
    EQR = enum.auto()


class PROJ_METHOD(enum.Enum):
    PSEUDO_AFINE = enum.auto()
    POLY_3DIM = enum.auto()


class INTERP_METHOD(enum.Enum):
    NEAREST_NEIGHBOR = enum.auto()
    BILINIEAR = enum.auto()


class Projection(ABC):
    pass


class ProjectionEQR(Projection):
    """
    EQuiRectangular projection
    """
    # WGS84
    WIDTH_KM = 40075.01668558
    HIGHT_KM = 40007.86193085 / 2
    # WIDTH_KM = 40000
    # HIGHT_KM = 20000

    PROJECTION_TYPE = PROJ_TYPE.EQR
    PROJECTION_NAME = PROJ_TYPE.EQR.name

    # ----------------------------
    # Public
    # ----------------------------
    def __init__(self, lat: np.ndarray, lon: np.ndarray, spatial_reso_m: float, map_area=None, quality='L', method: PROJ_METHOD=PROJ_METHOD.PSEUDO_AFINE):
        """

        :param lat:
        :param lon:
        :param spatial_reso_m:
        :param map_area:
        :param quality:
        :param method:
        """
        logging.debug('Initialize {0}'.format(self.__class__.__name__))

        # Define EQR map parameters according to user setting
        self.spatial_reso_km = spatial_reso_m / 1000.
        self.num_glb_w_pxls = int(self.WIDTH_KM / self.spatial_reso_km)
        self.num_glb_h_pxls = int(self.HIGHT_KM / self.spatial_reso_km)

        if map_area is None:
            self.per_area = [0, 1, 0, 1]
        elif abs(map_area[0] - map_area[1]) > 180:
            map_area = np.array(map_area, dtype=np.float64)
            map_area[0:2] = np.sort(np.mod(map_area[0:2], 360))
            self.per_area = np.concatenate(((map_area[0:2] + 180.) / 360., (map_area[2:] - 90.) / -180.))
        else:
            map_area = np.array(map_area, dtype=np.float64)
            self.per_area = np.concatenate(((map_area[0:2] + 180.) / 360., (map_area[2:] - 90.) / -180.))

        self.tgt_pxl_rect = np.array([self.num_glb_w_pxls, self.num_glb_w_pxls, self.num_glb_h_pxls, self.num_glb_h_pxls]) * self.per_area
        self.tgt_pxl_rect = [int(self.tgt_pxl_rect[0]), math.ceil(self.tgt_pxl_rect[1]), int(self.tgt_pxl_rect[2]), math.ceil(self.tgt_pxl_rect[3])]
        self.tgt_deg_rect = np.array([*self.EQR_x_pos2lon(self.tgt_pxl_rect[0:2]), *self.EQR_y_pos2lat(self.tgt_pxl_rect[2:])])

        self.proj_width = self.tgt_pxl_rect[1] - self.tgt_pxl_rect[0] + 1
        self.proj_height = self.tgt_pxl_rect[3] - self.tgt_pxl_rect[2] + 1
        self.x_offset = self.tgt_pxl_rect[0]
        self.y_offset = self.tgt_pxl_rect[2]

        logging.debug('  * Requested aera: {0}'.format(map_area))
        logging.debug('  * Spatial resolution: {0} km'.format(self.spatial_reso_km))
        logging.debug('  * Loc. of rect [West, East, North, South]: Deg. {0} <=> EQR pxl pos. {1}'.format(self.tgt_deg_rect, self.tgt_pxl_rect))
        logging.debug('  * Width: {0}, Height: {1}'.format(self.proj_width, self.proj_height))

        # cal. index for projection map
        pos_x_img = np.zeros((self.proj_height, self.proj_width), dtype=np.float32) * np.NaN
        pos_y_img = np.zeros((self.proj_height, self.proj_width), dtype=np.float32) * np.NaN

        lon = np.mod(lon, 360)
        x_idx_eqr = self.EQR_lon2x_pos(lon) - self.x_offset
        y_idx_eqr = self.EQR_lat2y_pos(lat) - self.y_offset

        (h, w) = lat.shape
        (x_idx_l1b, y_idx_l1b) = np.meshgrid(np.arange(0, w), np.arange(0, h))
        x_idx_l1b = x_idx_l1b
        y_idx_l1b = y_idx_l1b
        granularity = 25 # quality == 'H'
        if quality == 'L':
            granularity = 50
        elif quality == 'H':
            granularity = 10

        if spatial_reso_m == 250.:
            granularity = granularity * 4

        logging.debug('  * Proj. method: {0}'.format(method.name))
        if method == PROJ_METHOD.PSEUDO_AFINE:
            for h_x_idx_eqr, h_y_idx_eqr, h_x_idx_l1b, h_y_idx_l1b in zip(
                    np.array_split(x_idx_eqr, int(h/granularity), axis=0), np.array_split(y_idx_eqr, int(h/granularity), axis=0),
                    np.array_split(x_idx_l1b, int(h/granularity), axis=0), np.array_split(y_idx_l1b, int(h/granularity), axis=0)):
                for tgt_x_idx_eqr, tgt_y_idx_eqr, tgt_x_idx_l1b, tgt_y_idx_l1b in zip(
                        np.array_split(h_x_idx_eqr, int(w / granularity), axis=1), np.array_split(h_y_idx_eqr, int(w / granularity), axis=1),
                        np.array_split(h_x_idx_l1b, int(w / granularity), axis=1), np.array_split(h_y_idx_l1b, int(w / granularity), axis=1)):
                    (eqr2scene_coef_x, eqr2scene_coef_y) = self._cal_EQR2Scene_coef_for_pseudo_afine_trans(tgt_x_idx_eqr, tgt_y_idx_eqr, tgt_x_idx_l1b, tgt_y_idx_l1b)

                    # subset
                    x_range = [int(np.nanmin(tgt_x_idx_eqr)), int(np.ceil(np.nanmax(tgt_x_idx_eqr))) + 1]
                    y_range = [int(np.nanmin(tgt_y_idx_eqr)), int(np.ceil(np.nanmax(tgt_y_idx_eqr))) + 1]
                    (x_idx_ret, y_idx_ret) = np.meshgrid(np.arange(*x_range), np.arange(*y_range))
                    pos_x_img[y_idx_ret, x_idx_ret] = self._pseudo_afine_trans(eqr2scene_coef_x, x_idx_ret.astype(np.float64), y_idx_ret.astype(np.float64))
                    pos_y_img[y_idx_ret, x_idx_ret] = self._pseudo_afine_trans(eqr2scene_coef_y, x_idx_ret.astype(np.float64), y_idx_ret.astype(np.float64))

        elif method == PROJ_METHOD.POLY_3DIM:
            raise NotImplementedError()

        # Trim index on out of image range
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'invalid value encountered in (greater|less)')
            mask = np.where((pos_x_img < -0.5) | (pos_x_img > (w - 1)) | (pos_y_img < -0.5) | (pos_y_img > (h - 1)))
        pos_x_img[mask] = np.NaN
        pos_y_img[mask] = np.NaN

        self.valid_index = np.where(~np.isnan(pos_y_img))
        self.pos_x_img = pos_x_img[~np.isnan(pos_x_img)]
        self.pos_y_img = pos_y_img[~np.isnan(pos_y_img)]

    def run(self, data: np.ndarray, interpolation: INTERP_METHOD=INTERP_METHOD.NEAREST_NEIGHBOR):
        """

        :param data:
        :param interpolation:
        :return:
        """
        ret_img = np.zeros((self.proj_height, self.proj_width), dtype=np.float32) * np.NaN

        # data[np.isnan(data)] = 999
        if interpolation == INTERP_METHOD.NEAREST_NEIGHBOR:
            ret_img[self.valid_index] = data[(self.pos_y_img + 0.5).astype(np.int), (self.pos_x_img + 0.5).astype(np.int)]
        elif interpolation == INTERP_METHOD.BILINIEAR:
            raise NotImplementedError()

        return ret_img


    def _cal_EQR2Scene_coef_for_pseudo_afine_trans(self, tgt_x_idx_eqr: np.ndarray, tgt_y_idx_eqr: np.ndarray, tgt_x_idx_scene: np.ndarray, tgt_y_idx_scene: np.ndarray):
        x_eqr = tgt_x_idx_eqr.flatten()
        y_eqr = tgt_y_idx_eqr.flatten()
        x_scene = tgt_x_idx_scene.flatten()
        y_scene = tgt_y_idx_scene.flatten()

        a = np.array([x_eqr * y_eqr, x_eqr, y_eqr, np.ones(y_eqr.shape)]).T
        coef_x = np.linalg.solve(a.T @ a, a.T @ x_scene)
        coef_y = np.linalg.solve(a.T @ a, a.T @ y_scene)

        return coef_x, coef_y


    def _pseudo_afine_trans(self, params, x, y):
        return params[0] * y * x + params[1] * x + params[2] * y + params[3]

    def EQR_x_pos2lon(self, x_pos):
        return (np.array(x_pos, dtype=np.float64) / self.num_glb_w_pxls) * 360. - 180.

    def EQR_y_pos2lat(self, y_pos):
        return (np.array(y_pos, dtype=np.float64) / self.num_glb_h_pxls ) * -180. + 90.

    def EQR_lon2x_pos(self, lon):
        return ((np.array(lon, dtype=np.float64) + 180.) / 360.) * self.num_glb_w_pxls

    def EQR_lat2y_pos(self, lat):
        return ((np.array(lat, dtype=np.float64) - 90.) / -180.) * self.num_glb_h_pxls

    def get_latitude(self):
        y_indexes = np.arange(self.tgt_pxl_rect[2], self.tgt_pxl_rect[3]+1).reshape(-1, 1)
        y_indexes = np.tile(y_indexes, (1, self.proj_width))

        return self.EQR_y_pos2lat(y_indexes)

    def get_longitude(self):
        x_indexes = np.arange(self.tgt_pxl_rect[0], self.tgt_pxl_rect[1]+1).reshape(1, -1)
        x_indexes = np.tile(x_indexes, (self.proj_height, 1))

        return self.EQR_x_pos2lon(x_indexes)

    # ----------------------------
    # Private
    # ----------------------------

class ProjectionEQR4Tile(ProjectionEQR):
    def __init__(self, lat: np.ndarray, lon: np.ndarray, spatial_reso_m: float, htile: int, map_area=None, quality='L', method: PROJ_METHOD=PROJ_METHOD.PSEUDO_AFINE):
        self._cal_intermediate_data(lat, lon, htile)

        super().__init__(self.intermediate_lat, self.intermediate_lon, spatial_reso_m, map_area, quality, method)

    def run(self, data: np.ndarray, interpolation: INTERP_METHOD=INTERP_METHOD.NEAREST_NEIGHBOR):
        intermediate_img = np.zeros((self.intermediate_height, self.intermediate_width), dtype=np.float32) * np.NaN

        if interpolation == INTERP_METHOD.NEAREST_NEIGHBOR:
            intermediate_img[self.intermediate_valid_index] = data[(self.intermediate_pos_y_img).astype(np.int), (self.intermediate_pos_x_img).astype(np.int)]
        elif interpolation == INTERP_METHOD.BILINIEAR:
            raise NotImplementedError()

        data = super().run(intermediate_img, interpolation)

        return data

    def get_latitude(self):
        lat = super().get_latitude()

        return lat

    def get_longitude(self):
        lon = super().get_longitude()

        return lon

    def cal_lon_pos_on_equator(self, num_tile_pixels, lon):
        d = 180. / num_tile_pixels / 18
        np_0 = 2 * float(Decimal(180. / d).quantize(Decimal('0'), rounding=ROUND_HALF_UP))

        return (np_0 * lon / 360) + (np_0 / 2) - 0.5

    def _cal_lon_on_equator(self, num_tile_pixels, lon_pos):
        d = 180. / num_tile_pixels / 18
        np_0 = 2 * float(Decimal(180. / d).quantize(Decimal('0'), rounding=ROUND_HALF_UP))

        return (360. / np_0) * (lon_pos - (np_0 / 2) + 0.5)

    def _cal_lon_pos_from_np_i(self, num_tile_pixels, lon, np_i):
        d = 180. / num_tile_pixels / 18
        np_0 = 2 * float(Decimal(180. / d).quantize(Decimal('0'), rounding=ROUND_HALF_UP))

        return (np_i * lon / 360) + (np_0 / 2) - 0.5

    def cal_np_i(self, num_tile_pixels, lat):
        d = 180. / num_tile_pixels / 18
        np_0 = 2 * float(Decimal(180. / d).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
        np_i = (np_0 * np.cos(np.deg2rad(lat))).astype(np.float32).tolist()
        np_i = list(map(lambda e: int(Decimal(e).quantize(Decimal('0'), rounding=ROUND_HALF_UP)), np_i))
        return np.array(np_i, dtype=np.int32)

    def _cal_intermediate_data(self, lat, lon, htile):
        num_pixels = lat.shape[0]
        min_max_lon = np.concatenate(([np.nanmin(lon)], [np.nanmax(lon)]))
        min_max_lon_pos = self.cal_lon_pos_on_equator(num_pixels, min_max_lon).astype(int)
        x_pos = np.tile(np.arange(min_max_lon_pos[0], min_max_lon_pos[1]+1, 1), (lat.shape[0], 1))
        intermediate_lon = self._cal_lon_on_equator(num_pixels, x_pos)
        np_i = np.tile(self.cal_np_i(num_pixels, lat[:,0]).reshape(-1, 1), (1, intermediate_lon.shape[1]))
        intermediate_x_pos = self._cal_lon_pos_from_np_i(num_pixels, intermediate_lon, np_i) - htile * num_pixels
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'invalid value encountered in (greater_equal|less)')
            intermediate_x_pos[intermediate_x_pos < 0] = np.NaN
            intermediate_x_pos[intermediate_x_pos >= num_pixels] = np.NaN
        intermediate_y_pos = (np.ones(intermediate_x_pos.shape[0], dtype=np.int32) * np.arange(0, intermediate_x_pos.shape[0], dtype=np.int32)).reshape(-1, 1)
        intermediate_y_pos = np.tile(intermediate_y_pos, (1, intermediate_x_pos.shape[1]))

        self.intermediate_valid_index = np.where(~np.isnan(intermediate_x_pos))
        self.intermediate_pos_x_img = intermediate_x_pos[~np.isnan(intermediate_x_pos)]
        self.intermediate_pos_y_img = intermediate_y_pos[~np.isnan(intermediate_x_pos)]
        self.intermediate_lat = np.tile(lat[:,0].reshape(-1, 1), (1, intermediate_x_pos.shape[1]))
        self.intermediate_lon = intermediate_lon
        self.intermediate_height = intermediate_x_pos.shape[0]
        self.intermediate_width = intermediate_x_pos.shape[1]

# EOF
