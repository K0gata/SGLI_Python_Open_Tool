import numpy as np
import math
import enum
import logging
from abc import ABC, abstractmethod, abstractproperty
from decimal import Decimal, ROUND_HALF_UP
import matplotlib.pyplot as plt


class PROJ_TYPE(enum.Enum):
    SCENE = enum.auto()
    TILE = enum.auto()
    EQR = enum.auto()


class PROJ_METHOD(enum.Enum):
    PSEUDO_AFFINE = enum.auto()
    POLY_3DIM = enum.auto()

class PROJ_QUALITY(enum.Enum):
    HIGH = enum.auto()
    MODERATE = enum.auto()
    LOW = enum.auto()

class INTERP_METHOD(enum.Enum):
    NEAREST_NEIGHBOR = enum.auto()
    BILINIEAR = enum.auto()

class SPATILA_RESOLUTION_UNIT(enum.Enum):
    DEGREE = enum.auto()
    METER = enum.auto()

class Projection(ABC):
    pass


class ProjectionEQR(Projection):
    """
    EQuiRectangular projection
    """
    # ----------------------------
    # Class attribute
    # ----------------------------
    #
    # WIDTH_KM = 40000
    # HIGHT_KM = 20000
    # WGS84
    WIDTH_KM = 40075.01668558
    HIGHT_KM = 40007.86193085 / 2
    METER_PER_LON = WIDTH_KM * 1000. / 360.
    METER_PER_LAT = HIGHT_KM * 1000. / 180.

    PROJECTION_TYPE = PROJ_TYPE.EQR
    PROJECTION_NAME = PROJ_TYPE.EQR.name

    # ----------------------------
    # Public
    # ----------------------------
    def _set_projecting_coordinates(self, lat, lon, draw_map_area):
        self.num_glb_w_pxls = int(360. / self.spatial_resolution_deg)
        self.num_glb_h_pxls = int(180. / self.spatial_resolution_deg) + 1
        self.source_map_area = self._get_EQR_geo_rect(lat, lon)
        if draw_map_area is None:
            draw_map_area = self.source_map_area
        if draw_map_area[3] < draw_map_area[2]:
            draw_map_area[3] = 360. + draw_map_area[3]

        # Calculate a projecting position
        rel_draw_map_area = np.concatenate((((draw_map_area[0:2] - 90.) / -180.), (draw_map_area[2:] + 180.) / 360.))
        draw_pxl_rect = np.array([self.num_glb_h_pxls, self.num_glb_h_pxls, self.num_glb_w_pxls, self.num_glb_w_pxls]) * rel_draw_map_area
        draw_pxl_rect = [int(draw_pxl_rect[0]), math.ceil(draw_pxl_rect[1]), int(draw_pxl_rect[2]), math.ceil(draw_pxl_rect[3])]
        draw_deg_rect = np.array([*self.EQR_y_pos2lat(draw_pxl_rect[0:2]), *self.EQR_x_pos2lon(draw_pxl_rect[2:])])

        # When the same start and end point on the longitude.
        if (draw_map_area[2] == draw_map_area[3]) or ((draw_map_area[2] == -180.) and (draw_map_area[3] == 180.)):
            if draw_pxl_rect[2] >= self.num_glb_w_pxls:
                draw_pxl_rect[2] = draw_pxl_rect[2] % self.num_glb_w_pxls
            draw_pxl_rect[3] = draw_pxl_rect[2] + self.num_glb_w_pxls - 1
            draw_deg_rect[3] = self.EQR_x_pos2lon(draw_pxl_rect[3])

        # Set the parameters
        self.proj_height = draw_pxl_rect[1] - draw_pxl_rect[0] + 1
        self.proj_width = draw_pxl_rect[3] - draw_pxl_rect[2] + 1
        self.y_offset = draw_pxl_rect[0]
        self.x_offset = draw_pxl_rect[2]
        self.draw_deg_rect = draw_deg_rect
        self.draw_pxl_rect = draw_pxl_rect
        self.draw_map_area = draw_map_area

    def __init__(self, lat: np.ndarray, lon: np.ndarray,
                 spatial_resolution: float, spatial_resolution_unit: SPATILA_RESOLUTION_UNIT=SPATILA_RESOLUTION_UNIT.METER,
                 projection_method: PROJ_METHOD=PROJ_METHOD.PSEUDO_AFFINE, quality=PROJ_QUALITY.LOW, interpolation_method=INTERP_METHOD.NEAREST_NEIGHBOR,
                 draw_map_area: np.ndarray=None):
        """
        :param lat: (required)
        :param lon: (required)
        :param spatial_resolution: (required)
        :param spatial_resolution_unit: (required)
        :param projection_method: (optional, default: PROJ_METHOD.PSEUDO_AFINE)
        :param quality: (optional, default: PROJ_QUALITY.LOW)
        :param interpolation_method: (optional, default: INTERP_METHOD.NEAREST_NEIGHBOR)
        :param draw_map_area: (optional, default: inputted lat/lon ranges)
        """

        logging.debug('Initialize {0}'.format(self.__class__.__name__))

        # ===========================
        # Define EQR-map parameters
        # ===========================
        # Set spatial resolution
        self.spatial_resolution_deg = spatial_resolution
        if spatial_resolution_unit == SPATILA_RESOLUTION_UNIT.METER:
            logging.debug('  * Spatial resolution on the equator (m): {0} m'.format(spatial_resolution))
            self.spatial_resolution_deg = self.spatial_resolution_deg / self.METER_PER_LON
        self.source_spatial_resolution = spatial_resolution
        self.source_spatial_resolution_unit = spatial_resolution_unit
        logging.debug('  * Spatial resolution (deg): {0} deg.'.format(self.spatial_resolution_deg))

        # Calculate projecting coordinates
        self._set_projecting_coordinates(lat, lon, draw_map_area)

        logging.debug('  * Target projecting area [upper lat., lower lat., left lon., right lon.]: {0}'.format(self.draw_map_area))
        logging.debug('  * Output projecting area [deg], [pxl]: {0}, {1}'.format(self.draw_deg_rect, self.draw_pxl_rect))
        logging.debug('  * Projecting image size: h={0} lines, w={1} pixels'.format(self.proj_height, self.proj_width))

        # ===========================
        # Set flags and pre-processing
        # ===========================
        # if self.draw_map_area[2] == self.draw_map_area[3]:
        #     self._set_projecting_coordinates(lat, lon, np.array([*self.draw_map_area[:3], self.draw_map_area[3] + 360.], dtype=np.float64))
        #
        # if self.proj_width < 0:
        #     self._set_projecting_coordinates(lat, lon, np.array([*self.draw_map_area[:3], 360. + self.draw_map_area[3]], dtype=np.float64))

        source_pxl_rect_lon = self.EQR_lon2x_pos(self.source_map_area[2:])
        # self.is_split_source = True if (source_pxl_rect_lon[0] > source_pxl_rect_lon[-1]) else False
        self.is_split_source = True if (source_pxl_rect_lon[0] < self.draw_pxl_rect[2]) else False
        if self.is_split_source:
            logging.debug('  * Input data is divided by output position')
            self._set_projecting_coordinates(lat, lon, np.array([*self.draw_map_area[:2], *self.source_map_area[2:]], dtype=np.float64))

        # self.is_stride_over_180_lon = True if (self.draw_map_area[3] > 180.) else False
        # if self.is_stride_over_180_lon:
        #     lon[lon < 0] = 360. + lon[lon < 0]
        #     logging.debug('  * Projecting data is stride over +/-180 degree on the longitude.')

        if self.draw_map_area[3] > 180:
            lon[lon < 0] = 360. + lon[lon < 0]
            logging.debug('  * Projecting data is stride over +/-180 degree on the longitude.')

        # ===========================
        # Run Forward transformation
        # ===========================
        logging.debug('  * Run forward transformation')

        # Run
        source_y_idx_eqr = self.EQR_lat2y_pos(lat)
        source_x_idx_eqr = self.EQR_lon2x_pos(lon)

        # Drop the outer range of the projection
        proj_outer_area = (source_y_idx_eqr < self.draw_pxl_rect[0]) | (source_y_idx_eqr > self.draw_pxl_rect[1]) |\
                          (source_x_idx_eqr < self.draw_pxl_rect[2]) | (source_x_idx_eqr > self.draw_pxl_rect[3])

        source_y_idx_eqr[proj_outer_area] = np.NaN
        source_x_idx_eqr[proj_outer_area] = np.NaN
        source_y_idx_eqr = source_y_idx_eqr - self.y_offset
        source_x_idx_eqr = source_x_idx_eqr - self.x_offset

        # ===========================
        # Run backward transformation
        # ===========================
        logging.debug('  * Run backward transformation')
        logging.debug('    * Projection method: {0}'.format(projection_method.name))
        logging.debug('    * Projection quality: {0}'.format(quality.name))

        # Set calculating granularity of backward transformation
        granularity = 25 # PROJ_QUALITY.MODERATE
        if quality == PROJ_QUALITY.LOW:
            granularity = 50
        elif quality == PROJ_QUALITY.HIGH:
            granularity = 10

        # Set projection methods
        if projection_method == PROJ_METHOD.PSEUDO_AFFINE:
            selected_cal_coef_method = getattr(self, '_cal_coef_for_pseudo_affine_trans')
            selected_projection_method = getattr(self, '_pseudo_affine_trans')

        # Allocate positional arrays for projecting image
        pos_x_img = np.zeros((self.proj_height, self.proj_width), dtype=np.float32) * np.NaN
        pos_y_img = np.zeros((self.proj_height, self.proj_width), dtype=np.float32) * np.NaN

        # Generate index arrays for source data
        (source_h, source_w) = lat.shape
        (source_x_idx, source_y_idx) = np.meshgrid(np.arange(0, source_w), np.arange(0, source_h))

        # logging.info('{0} {1} {2} '.format(granularity, int(source_h/granularity), int(source_w / granularity)))

        # Run backward transformation
        for h_x_idx_eqr, h_y_idx_eqr, h_x_idx, h_y_idx, h_proj_outer_area in zip(
                np.array_split(source_x_idx_eqr, int(source_h/granularity), axis=0), np.array_split(source_y_idx_eqr, int(source_h/granularity), axis=0),
                np.array_split(source_x_idx, int(source_h/granularity), axis=0), np.array_split(source_y_idx, int(source_h/granularity), axis=0),
                np.array_split(proj_outer_area, int(source_h/granularity), axis=0)):

            # Skip this block because it was all composed of outer data.
            if h_proj_outer_area.all():
                continue

            for tgt_x_idx_eqr, tgt_y_idx_eqr, tgt_x_idx, tgt_y_idx, tgt_proj_outer_area in zip(
                    np.array_split(h_x_idx_eqr, int(source_w / granularity), axis=1), np.array_split(h_y_idx_eqr, int(source_w / granularity), axis=1),
                    np.array_split(h_x_idx, int(source_w / granularity), axis=1), np.array_split(h_y_idx, int(source_w / granularity), axis=1),
                    np.array_split(h_proj_outer_area, int(source_w / granularity), axis=1)):

                # Skip this block because it was all composed of outer data.
                if tgt_proj_outer_area.all():
                    continue

                # Calculate transformation matrices on x- and y-coordinates
                (eqr2scene_coef_x, eqr2scene_coef_y) = selected_cal_coef_method(
                    tgt_x_idx_eqr[~tgt_proj_outer_area], tgt_y_idx_eqr[~tgt_proj_outer_area],
                    tgt_x_idx[~tgt_proj_outer_area], tgt_y_idx[~tgt_proj_outer_area])
                x_range = [int(np.nanmin(tgt_x_idx_eqr)), int(np.ceil(np.nanmax(tgt_x_idx_eqr))) + 1]
                y_range = [int(np.nanmin(tgt_y_idx_eqr)), int(np.ceil(np.nanmax(tgt_y_idx_eqr))) + 1]
                (x_idx_ret, y_idx_ret) = np.meshgrid(np.arange(*x_range), np.arange(*y_range))
                pos_x_img[y_idx_ret, x_idx_ret] = selected_projection_method(eqr2scene_coef_x, x_idx_ret.astype(np.float64), y_idx_ret.astype(np.float64))
                pos_y_img[y_idx_ret, x_idx_ret] = selected_projection_method(eqr2scene_coef_y, x_idx_ret.astype(np.float64), y_idx_ret.astype(np.float64))

        # Trim index on out of image range
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'invalid value encountered in (greater|less)')
            mask = np.where((pos_x_img < -0.5) | (pos_x_img > (source_w - 1)) | (pos_y_img < -0.5) | (pos_y_img > (source_h - 1)))
        pos_x_img[mask] = np.NaN
        pos_y_img[mask] = np.NaN

        valid_idx_y, valid_idx_x = np.where(~np.isnan(pos_y_img))
        # Shift x-position when the source data is divided.
        if self.is_split_source is True:
            valid_idx_x = valid_idx_x + self.x_offset
            # Get the truth projection parameters
            self._set_projecting_coordinates(lat, lon, draw_map_area)
            valid_idx_x = (valid_idx_x - self.x_offset) % self.num_glb_w_pxls

        inner_x_idx = np.where(valid_idx_x < self.proj_width)
        self.valid_index = (valid_idx_y[inner_x_idx], valid_idx_x[inner_x_idx])
        self.pos_x_img = pos_x_img[~np.isnan(pos_x_img)][inner_x_idx]
        self.pos_y_img = pos_y_img[~np.isnan(pos_y_img)][inner_x_idx]

    def run(self, data: np.ndarray, interpolation: INTERP_METHOD=INTERP_METHOD.NEAREST_NEIGHBOR):
        """

        :param data:
        :param interpolation:
        :return:
        """
        ret_img = np.zeros((self.proj_height, self.proj_width), dtype=np.float32) * np.NaN

        # data[np.isnan(data)] = 999
        if interpolation == INTERP_METHOD.NEAREST_NEIGHBOR:
            y_pos = (self.pos_y_img + 0.5).astype(np.int)
            x_pos = (self.pos_x_img + 0.5).astype(np.int)
            # if self.is_split_source:
                # valid_idx_x = self.valid_index[1]
                # self.valid_index = (self.valid_index[0], valid_idx_x)
                # x_pos[x_pos >= self.draw_pxl_rect[3]] = x_pos[x_pos >= self.draw_pxl_rect[3]] - self.draw_pxl_rect[3]
            ret_img[self.valid_index] = data[y_pos, x_pos]
        elif interpolation == INTERP_METHOD.BILINIEAR:
            raise NotImplementedError()

        return ret_img

    def EQR_x_pos2lon(self, x_pos):
        return (np.array(x_pos, dtype=np.float64) / self.num_glb_w_pxls) * 360. - 180.

    def EQR_y_pos2lat(self, y_pos):
        return (np.array(y_pos, dtype=np.float64) / self.num_glb_h_pxls ) * -180. + 90.

    def EQR_lon2x_pos(self, lon):
        return ((np.array(lon, dtype=np.float64) + 180.) / 360.) * self.num_glb_w_pxls

    def EQR_lat2y_pos(self, lat):
        return ((np.array(lat, dtype=np.float64) - 90.) / -180.) * self.num_glb_h_pxls

    def get_latitude(self):
        y_indexes = np.arange(self.draw_pxl_rect[0], self.draw_pxl_rect[1]+1).reshape(-1, 1)
        y_indexes = np.tile(y_indexes, (1, self.proj_width))

        return self.EQR_y_pos2lat(y_indexes)

    def get_longitude(self):
        x_indexes = np.arange(self.draw_pxl_rect[2], self.draw_pxl_rect[3]+1).reshape(1, -1)
        x_indexes = np.tile(x_indexes, (self.proj_height, 1))

        return self.EQR_x_pos2lon(x_indexes)

    # ----------------------------
    # Private
    # ----------------------------
    def _get_EQR_geo_rect(self, lat, lon, over_180lon=False):
        """
        :param lat: latitude [deg.]
        :param lon: longitude [deg.]
        :return: [upper_lat lower_lat, left_lon, right_lon] (The Longitude ranges from -180 to 180 degrees)
        """
        left_lon = np.nanmin(lon)
        right_lon = np.nanmax(lon)
        if (left_lon - right_lon) < -180.:
            positive_lon = np.mod(lon, 360)
            lon_range = np.array([np.nanmin(positive_lon), np.nanmax(positive_lon)], dtype=np.float32)
            if over_180lon is False:
                lon_range[lon_range > 180] = lon_range[lon_range > 180] - 360
            (left_lon, right_lon) = lon_range
        rect = [np.nanmax(lat), np.nanmin(lat), left_lon, right_lon]

        return np.array(rect, np.float64)

    # Pseudo affine translation
    def _pseudo_affine_trans(self, params, x, y):
        return params[0] * y * x + params[1] * x + params[2] * y + params[3]

    def _cal_coef_for_pseudo_affine_trans(self, tgt_x_idx_eqr: np.ndarray, tgt_y_idx_eqr: np.ndarray, tgt_x_idx_scene: np.ndarray, tgt_y_idx_scene: np.ndarray):
        x_eqr = tgt_x_idx_eqr.flatten()
        y_eqr = tgt_y_idx_eqr.flatten()
        x_scene = tgt_x_idx_scene.flatten()
        y_scene = tgt_y_idx_scene.flatten()

        a = np.array([x_eqr * y_eqr, x_eqr, y_eqr, np.ones(y_eqr.shape)]).T
        coef_x = np.linalg.solve(a.T @ a, a.T @ x_scene)
        coef_y = np.linalg.solve(a.T @ a, a.T @ y_scene)

        return coef_x, coef_y


class ProjectionEQR4Tile(ProjectionEQR):
    def __init__(self, lat: np.ndarray, lon: np.ndarray, spatial_resolution: float,  htile: int, spatial_resolution_unit: SPATILA_RESOLUTION_UNIT=SPATILA_RESOLUTION_UNIT.METER, map_area=None, quality='L', method: PROJ_METHOD=PROJ_METHOD.PSEUDO_AFFINE):
        self._cal_intermediate_data(lat, lon, htile)

        super().__init__(self.intermediate_lat, self.intermediate_lon, spatial_resolution=spatial_resolution, spatial_resolution_unit=spatial_resolution_unit, map_area=map_area, quality=quality, method=method)

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
