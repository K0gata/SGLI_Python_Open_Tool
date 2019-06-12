import os
import h5py
import logging
import numpy as np
import warnings
import datetime
import time

from spot.config import SpotWarnings, TZ_UTC0, PROJ_TYPE
from spot.level1 import VNRL1B, IRSL1B
from spot.level2 import OceanL2, LandL2, AtmosphereL2, CryosphereL2, CryosphereOkhotskL2, RadianceL2
from spot.projection import ProjectionEQR, ProjectionEQR4Tile, PROJ_METHOD, INTERP_METHOD

# =============================
#  Function
# =============================


def tai93toGMT(tai93_time:float):
    tai93_epoc = time.mktime(time.strptime("UTC19930101", "%Z%Y%m%d")) - time.timezone
    gmt_time = time.gmtime(tai93_time + tai93_epoc)

    return datetime.datetime(*gmt_time[0:6], tzinfo=TZ_UTC0)

# =============================
#  File Class
# =============================


class File:
    # -----------------------
    # Constructor
    # -----------------------
    def __init__(self, filepath:str):
        # Set file path
        self.filepath = filepath
        self.filename = os.path.basename(filepath)

        # Open requested HDF5 file
        logging.debug('Open: {0}'.format(filepath))
        self.h5_file = None
        self.h5_file = h5py.File(filepath, 'r', swmr=True)

        # Load parameters
        glbl_attr_grp_attrs = self.h5_file['Global_attributes'].attrs
        self.product_level = glbl_attr_grp_attrs['Product_level'][0].decode('UTF-8')
        self.product_name = glbl_attr_grp_attrs['Product_name'][0].decode('UTF-8')
        self.product_version = glbl_attr_grp_attrs['Product_version'][0].decode('UTF-8')
        self.parameter_version = glbl_attr_grp_attrs['Parameter_version'][0].decode('UTF-8')
        self.algorithm_version = glbl_attr_grp_attrs['Algorithm_version'][0].decode('UTF-8')

        # Distinguish product type and generate a dedicated reader
        self.product_id = os.path.splitext(self.filename)[0][-10:-6]
        logging.debug(' * Product: {0}'.format(self.product_id))
        if self.product_id == 'VNRD' or self.product_id == 'VNRN':
            self._reader = VNRL1B(self.h5_file, self.product_id)
        elif self.product_id == 'IRSD' or self.product_id == 'IRSN':
            self._reader = IRSL1B(self.h5_file, self.product_id)
        elif self.product_id == 'NWLR' or self.product_id == 'IWPR' or self.product_id == 'SSTD' or \
                self.product_id == 'SSTN':
            self._reader = OceanL2(self.h5_file, self.product_id)
        elif self.product_id == 'SIPR':
            self._reader = CryosphereL2(self.h5_file, self.product_id)
        elif self.product_id == 'LAI_' or self.product_id == 'RSRF' or self.product_id == 'VGI_' or \
                self.product_id == 'AGB_' or self.product_id == 'LST_':
            self._reader = LandL2(self.h5_file, self.product_id)
        elif self.product_id == 'ARNP' or self.product_id == 'ARPL' or self.product_id == 'CLFG' or \
                self.product_id == 'CLPR' :
            self._reader = AtmosphereL2(self.h5_file, self.product_id)
        elif self.product_id == 'LTOA':
            self._reader = RadianceL2(self.h5_file, self.product_id)
        else:
            warnings.warn('{0} is not yet supported!'.format(self.product_id), SpotWarnings, stacklevel=2)
            raise NotSupportedError('{0} is not yet supported!'.format(self.product_id))

        # Initialize projection property
        self._projector = None
        self.original_projection_type = self._reader.PROJECTION_TYPE
        self.projection_type = self.original_projection_type
        self.corner_coordinate = None

        # Initialize cache spaces
        self.img_data_cache = {}
        self.img_projected_data_cache = {}
        for key in self.get_product_data_list():
            self.img_data_cache[key] = None
            self.img_projected_data_cache[key] = None

        self.geo_data_cache = {}
        self.geo_projected_data_cache = {}
        for key in self.get_geometry_data_list():
            self.geo_data_cache[key] = None
            self.geo_projected_data_cache[key] = None

    # -----------------------
    # Getter
    # -----------------------
    def get_product_data(self, prod_name: str, projection: str='auto'):
        """

        :param prod_name:
        :param projection:
            'auto': use default projection type
            'original': return original data from product file
        :return:
        """
        # Check requested product name
        if prod_name not in self.img_data_cache.keys():
            warnings.warn('{0} not found on {1}'.format(prod_name, self.filepath), SpotWarnings, stacklevel=2)
            return None

        # Detect projection type
        if projection == 'auto':
            tgt_proj_type = self.projection_type
        elif projection == 'original':
            tgt_proj_type = self.original_projection_type
        else:
            if projection not in self.get_allow_projection_type():
                warnings.warn('\'{0}\' is not allowed as projection type!　(type: {1})'.format(
                    projection, self.get_allow_projection_type()), SpotWarnings, stacklevel=2)
                return None
            tgt_proj_type = projection

        # Return cached data if cache has the requested data
        if (tgt_proj_type == self.original_projection_type) and (self.img_data_cache[prod_name] is not None):
            logging.debug('Use cache: {0} (original)'.format(prod_name))
            return self.img_data_cache[prod_name].copy()
        elif (self.projection_type == tgt_proj_type) and (self.img_projected_data_cache[prod_name] is not None):
            logging.debug('Use cache: {0} ({1})'.format(prod_name, tgt_proj_type))
            return self.img_projected_data_cache[prod_name].copy()

        # Get raw data
        if self.img_data_cache[prod_name] is not None:
            data = self.img_data_cache[prod_name].copy()
        else:
            data = self._reader.get_product_data(prod_name)
            # Set the data into cache space
            self.img_data_cache[prod_name] = data
            logging.debug('Read: {0} (original)'.format(prod_name))

        # Project the requested type if the requested type is not original projection type
        if (self.original_projection_type != tgt_proj_type) and (tgt_proj_type == self.projection_type):
            data = self._projector.run(data, INTERP_METHOD.NEAREST_NEIGHBOR)
            self.img_projected_data_cache[prod_name] = data
            logging.debug('Read: {0} ({1})'.format(prod_name, tgt_proj_type))
        elif (self.original_projection_type != tgt_proj_type) and (tgt_proj_type == PROJ_TYPE.EQR.name):
            lat = self.get_geometry_data('Latitude', projection='original')
            lon = self.get_geometry_data('Longitude', projection='original')
            if self.original_projection_type == PROJ_TYPE.SCENE.name:
                projector = ProjectionEQR(lat, lon, spatial_reso_m=self._reader.img_spatial_reso, map_area=self._get_corner_latlon())
            elif self.original_projection_type == PROJ_TYPE.TILE.name:
                projector = ProjectionEQR4Tile(lat, lon, spatial_reso_m=self._reader.img_spatial_reso, vtile=self._reader.vtile, map_area=self._get_corner_latlon())
            data = projector.run(data, INTERP_METHOD.NEAREST_NEIGHBOR)
            logging.debug('Read: {0} ({1})'.format(prod_name, tgt_proj_type))

        return data.copy()

    def get_unit(self, prod_name:str):
        # Check requested product name
        if prod_name in self.img_data_cache.keys() or prod_name in self.geo_data_cache.keys():
            return self._reader.get_unit(prod_name)

        warnings.warn('{0} not found on {1}'.format(prod_name, self.filepath), SpotWarnings, stacklevel=2)
        return None

    def get_flag(self, flags=[], projection: str='auto'):
        """

        :param args: int (0-15)
        :return: True is flagged pixel.
        """
        # Validation
        if len(flags) < 1:
            warnings.warn('Argument is empty!', SpotWarnings, stacklevel=2)
            return None
        for arg in flags:
            if type(arg) is not int:
                warnings.warn('Arguments allow only int type: {0}({1})'.format(arg, type(arg)), SpotWarnings, stacklevel=2)
                return None

        # Detect projection type
        if projection == 'auto':
            tgt_proj_type = self.projection_type
        elif projection == 'original':
            tgt_proj_type = self.original_projection_type
        else:
            if projection not in self.get_allow_projection_type():
                warnings.warn('\'{0}\' is not allowed as projection type!　(type: {1})'.format(
                    projection, self.get_allow_projection_type()), SpotWarnings, stacklevel=2)
                return None
            tgt_proj_type = projection

        # Get flags
        flag_val = np.sum(np.power(2, np.array(flags, dtype=np.uint32)))
        flag_data = self.get_product_data('QA_flag', 'original')
        flag_data = np.bitwise_and(flag_data, flag_val).astype(np.bool)

        # project the requested type if the requested type is not original projection type
        if (self.original_projection_type != tgt_proj_type) and (tgt_proj_type == self.projection_type):
            flag_data = self._projector.run(flag_data, INTERP_METHOD.NEAREST_NEIGHBOR)
            logging.debug('Read: {0} ({1})'.format('QA_flag', tgt_proj_type))
        elif (self.original_projection_type != tgt_proj_type) and (tgt_proj_type == PROJ_TYPE.EQR.name):
            lat = self.get_geometry_data('Latitude', projection='original')
            lon = self.get_geometry_data('Longitude', projection='original')
            if self.original_projection_type == PROJ_TYPE.SCENE.name:
                projector = ProjectionEQR(lat, lon, spatial_reso_m=self._reader.img_spatial_reso, map_area=self._get_corner_latlon())
            elif self.original_projection_type == PROJ_TYPE.TILE.name:
                projector = ProjectionEQR4Tile(lat, lon, spatial_reso_m=self._reader.img_spatial_reso, htile=self._reader.htile, map_area=self._get_corner_latlon())
            flag_data = projector.run(flag_data, INTERP_METHOD.NEAREST_NEIGHBOR)
            logging.debug('Read: {0} ({1})'.format('QA_flag', tgt_proj_type))

        flag_data[np.isnan(flag_data)] = 0
        return np.array(flag_data, dtype=np.bool)

    def get_geometry_data(self, data_name: str, projection: str='auto', **kwargs):
        """
        :param data_name: a dataset name in Geometry data group
        Options:
            :param projection:
                'auto': use default projection type
                'original': return original data from product file

            Level-2 scene product)
                :param interval:
                    None or 'none: no interpolation (= source data size).
                    'auto': interpolation with defined grid_interval value.
                    integer interpolation with use requested interval value.
                :param fit_img_size: True or False
        :return:
        """
        # Check requested product name
        if data_name not in self.geo_data_cache.keys():
            warnings.warn('{0} not found on {1}'.format(data_name, self.filepath), SpotWarnings, stacklevel=2)
            return None

        # Detect projection type
        if projection == 'auto':
            tgt_proj_type = self.projection_type
        elif projection == 'original':
            tgt_proj_type = self.original_projection_type
        else:
            if projection not in self.get_allow_projection_type():
                warnings.warn('\'{0}\' is not allowed as projection type!　(type: {1})'.format(
                    projection, self.get_allow_projection_type()), SpotWarnings, stacklevel=2)
                return None
            tgt_proj_type = projection

        # Parse keyword arguments:
        if tgt_proj_type == PROJ_TYPE.SCENE.name or self.original_projection_type == PROJ_TYPE.SCENE.name:
            scene_args = {'interval': 'auto', 'fit_img_size': True}
            if tgt_proj_type == PROJ_TYPE.SCENE.name:
                for key in kwargs.keys():
                    if key == 'interval' or key == 'fit_img_size':
                        scene_args[key] = kwargs[key]

        # Return cached data if cache has the requested data
        if (tgt_proj_type == self.original_projection_type) and (self.geo_data_cache[data_name] is not None):
            # Scene with auto interval and fitted image size
            if (tgt_proj_type == PROJ_TYPE.SCENE.name) and (scene_args['interval'] == 'auto') and (scene_args['fit_img_size'] is True):
                logging.debug('Use cache: {0} (original)'.format(data_name))
                return self.geo_data_cache[data_name].copy()
            # Tile
            elif tgt_proj_type == PROJ_TYPE.TILE.name:
                logging.debug('Use cache: {0} (original)'.format(data_name))
                return self.geo_data_cache[data_name].copy()
        elif (self.projection_type == tgt_proj_type) and (self.geo_projected_data_cache[data_name] is not None):
            logging.debug('Use cache: {0} ({1})'.format(data_name, tgt_proj_type))
            return self.geo_projected_data_cache[data_name].copy()

        # Get raw data
        if self.geo_data_cache[data_name] is not None:
            data = self.geo_data_cache[data_name].copy()
        elif self.original_projection_type == PROJ_TYPE.SCENE.name:
            data = self._reader.get_geometry_data(data_name, interval=scene_args['interval'], fit_img_size=scene_args['fit_img_size'])
            # Set the data into cache space
            (data_size_lin, data_size_pxl) = data.shape
            if (scene_args['fit_img_size'] is True) and (self._reader.img_n_lin <= data_size_lin) and (self._reader.img_n_pix <= data_size_pxl):
                self.geo_data_cache[data_name] = data
            logging.debug('Read: {0} (original)'.format(data_name))
        elif self.original_projection_type == PROJ_TYPE.TILE.name:
            data = self._reader.get_geometry_data(data_name, interval='auto', fit_img_size=False)
            # Set the data into cache space
            self.geo_data_cache[data_name] = data
            logging.debug('Read: {0} (original)'.format(data_name))

        # Project the requested type if the requested type is not original projection type
        if (self.original_projection_type != tgt_proj_type) and (tgt_proj_type == self.projection_type):
            if data_name == 'Latitude':
                data = self._projector.get_latitude()
            elif data_name == 'Longitude':
                data = self._projector.get_longitude()
                data[data > 180] = data[data > 180] - 360
            else:
                data = self._projector.run(data, INTERP_METHOD.NEAREST_NEIGHBOR)

            self.geo_projected_data_cache[data_name] = data
            logging.debug('Read: {0} ({1})'.format(data_name, tgt_proj_type))

        elif (self.original_projection_type != tgt_proj_type) and (tgt_proj_type == PROJ_TYPE.EQR.name):
            lat = self.get_geometry_data('Latitude', projection='original')
            lon = self.get_geometry_data('Longitude', projection='original')
            if self.original_projection_type == PROJ_TYPE.SCENE.name:
                projector = ProjectionEQR(lat, lon, spatial_reso_m=self._reader.img_spatial_reso, map_area=self._get_corner_latlon())
            elif self.original_projection_type == PROJ_TYPE.TILE.name:
                projector = ProjectionEQR4Tile(lat, lon, spatial_reso_m=self._reader.img_spatial_reso, h=self._reader.htile, map_area=self._get_corner_latlon())

            if data_name == 'Latitude':
                data = projector.get_latitude()
            elif data_name == 'Longitude':
                data = projector.get_longitude()
            else:
                data = projector.run(data, INTERP_METHOD.NEAREST_NEIGHBOR)
            logging.debug('Read: {0} ({1})'.format(data_name, tgt_proj_type))

        return data.copy()

    def get_product_data_list(self):
        return self._reader.get_product_data_list()

    def get_geometry_data_list(self):
        return self._reader.get_geometry_data_list()

    def get_projection_type(self):
        return self.projection_type

    # -----------------------
    # Search methods
    # -----------------------
    def search_pos_from_latlon(self, tgt_lat:float, tgt_lon:float):
        lat_data = self.get_geometry_data('Latitude')
        lon_data = self.get_geometry_data('Longitude')

        lat_distance = lat_data - tgt_lat
        lon_distance = lon_data - tgt_lon

        # Check lat/lon ranges
        is_lat_in_range = any((lat_distance[0, :] * lat_distance[-1, 0]) < 0)
        is_lon_in_range = any((lon_distance[:, 0] * lon_distance[:, -1]) < 0)
        if not (is_lat_in_range and is_lon_in_range):
            warnings.warn('Out of image!'.format(self.product_id), SpotWarnings, stacklevel=2)
            return None

        # Search requested position
        distance = lat_distance * lat_distance + lon_distance * lon_distance

        return np.unravel_index(distance.argmin(), distance.shape)

    # def seatch_pos_from_rect(self, tgt_u_lat:float, tgt_u_lon:float, tgt_l_lat:float, tgt_l_lon:float):
    #     warnings.warn('Not implemented!')
    #     pass

    # -----------------------
    # Projection methods
    # -----------------------
    def get_allow_projection_type(self):
        return self._reader.get_allow_projection_type()

    def get_current_projection_type(self):
        return self.projection_type

    def set_projection_type(self, projection: str):
        if self.projection_type == projection:
            return True

        if projection not in self.get_allow_projection_type():
            warnings.warn('\'{0}\' is not allowed as projection type!　(type: {1})'.format(
                projection, self.get_allow_projection_type()), SpotWarnings, stacklevel=2)
            return False

        self.clean_projection_cache()

        if projection == self.original_projection_type:
            self.projection_type = self.original_projection_type
            self._projector = None
            return True

        elif (projection == PROJ_TYPE.EQR.name) and (self.original_projection_type == PROJ_TYPE.SCENE.name):
            lat = self.get_geometry_data('Latitude', projection='original')
            lon = self.get_geometry_data('Longitude', projection='original')
            self._projector = ProjectionEQR(lat, lon, spatial_reso_m=self._reader.img_spatial_reso, map_area=self._get_corner_latlon())
            self.projection_type = self._projector.PROJECTION_NAME
        elif (projection == PROJ_TYPE.EQR.name) and (self.original_projection_type == PROJ_TYPE.TILE.name):
            lat = self.get_geometry_data('Latitude', projection='original')
            lon = self.get_geometry_data('Longitude', projection='original')
            self._projector = ProjectionEQR4Tile(lat, lon, spatial_reso_m=self._reader.img_spatial_reso, htile=self._reader.htile, map_area=self._get_corner_latlon())
            self.projection_type = self._projector.PROJECTION_NAME

    def _get_corner_latlon(self):
        if self.corner_coordinate is None:
            lat = self.get_geometry_data('Latitude', projection='original')
            lon = self.get_geometry_data('Longitude', projection='original')
            positive_lon = np.mod(lon, 360)
            lon_range = np.array([np.nanmin(positive_lon), np.nanmax(positive_lon)], dtype=np.float32)
            lon_range[lon_range > 180] = lon_range[lon_range > 180] - 360
            self.corner_coordinate = [*lon_range, np.nanmax(lat), np.nanmin(lat)]
        return self.corner_coordinate

    # -----------------------
    # File utility
    # -----------------------
    def close(self):
        if self.h5_file is not None:
            logging.debug('Close: {0}'.format(self.filepath))
            self.h5_file.close()

    # -----------------------
    # Cache utility
    # -----------------------
    def clean_cache(self):
        for key in self.img_data_cache.keys():
            self.img_data_cache[key] = None

        for key in self.geo_data_cache.keys():
            self.geo_data_cache[key] = None

    def clean_projection_cache(self):
        for key in self.img_data_cache.keys():
            self.img_projected_data_cache[key] = None

        for key in self.geo_data_cache.keys():
            self.geo_projected_data_cache[key] = None

    def clean_all_cache(self):
        self.clean_cache()
        self.clean_projection_cache()

    # +++++++++++++++++++++++
    # Private methods
    # +++++++++++++++++++++++


# =============================
#  Exception Class
# =============================


class NotSupportedError(Exception):
    pass
