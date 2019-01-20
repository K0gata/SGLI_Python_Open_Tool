import os
import h5py
import logging
import numpy as np
import warnings
import datetime
import time

from spot.config import SpotWarnings, TZ_UTC0
from spot.level2 import OceanL2, LandL2, AtmosphereL2, CryosphereL2, CryosphereOkhotskL2

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
        if self.product_id == 'NWLR' or self.product_id == 'IWPR':
            self._reader = OceanL2(self.h5_file)
        elif self.product_id == 'SIPR':
            self._reader = CryosphereL2(self.h5_file)
        elif self.product_id == 'LAI_':
            self._reader = LandL2(self.h5_file)
        elif self.product_id == 'ARNP':
            self._reader = AtmosphereL2(self.h5_file)
        else:
            warnings.warn('{0} is not yet supported!'.format(self.product_id), SpotWarnings, stacklevel=2)
            raise NotSupportedError('{0} is not yet supported!'.format(self.product_id))

        # Initialize cache spaces
        self.img_data_cache = {}
        for key in self.get_product_list():
            self.img_data_cache[key] = None

        self.geo_data_cache = {}
        for key in self.get_geometry_data_list():
            self.geo_data_cache[key] = None

    # -----------------------
    # Getter
    # -----------------------
    def get_product(self, prod_name:str):
        # Check requested product name
        if prod_name not in self.img_data_cache.keys():
            warnings.warn('{0} not found on {1}'.format(prod_name, self.filepath), SpotWarnings, stacklevel=2)
            return None

        # Return cached data if cache has the requested data
        if self.img_data_cache[prod_name] is not None:
            logging.debug('Use cache: {0}'.format(prod_name))
            return self.img_data_cache[prod_name].copy()

        # Get data
        data = self._reader.get_product(prod_name)

        # Set the data into cache space
        self.img_data_cache[prod_name] = data

        return data.copy()

    def get_unit(self, prod_name:str):
        # Check requested product name
        if prod_name not in self.img_data_cache.keys():
            warnings.warn('{0} not found on {1}'.format(prod_name, self.filepath), SpotWarnings, stacklevel=2)
            return None

        # Get attrs set
        unit_name = 'Unit'
        real_prod_name = prod_name
        if 'Rrs' in prod_name:
            real_prod_name = prod_name.replace('Rrs', 'NWLR')
            unit_name = 'Rrs_unit'
        attrs = self.h5_file['/Image_data/' + real_prod_name].attrs

        # Get unit
        if unit_name not in attrs:
            return ''
        return attrs[unit_name][0].decode('UTF-8')

    def get_flag(self, *args):
        """

        :param args: int (0-15)
        :return: True is flagged pixel.
        """
        # Validation
        if len(args) < 1:
            warnings.warn('Argument is empty!', SpotWarnings, stacklevel=2)
            return None
        for arg in args:
            if type(arg) is not int:
                warnings.warn('Arguments allow only int type: {0}({1})'.format(arg, type(arg)), SpotWarnings, stacklevel=2)
                return None

        # Get flags
        flag_val = np.sum(np.power(2, np.array(args, dtype=np.uint32)))
        flag_data = self.get_product('QA_flag')
        flag_data = np.bitwise_and(flag_data, flag_val).astype(np.bool)

        return flag_data

    #def get_geometry_data(self, data_name:str, interval='auto', fit_img_size=True):
    def get_geometry_data(self, data_name: str, **kwargs):
        """
        :param data_name: a dataset name in Geometry data group
        Options:
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

        # Parse keyword arguments:
        if self._reader.projection_type == 'Scene':
            scene_args = {'interval': 'auto', 'fit_img_size': True}
            for key in kwargs.keys():
                if key == 'interval' or key == 'fit_img_size':
                    scene_args[key] = kwargs[key]

        # Return cached data if cache has the requested data
        if self.geo_data_cache[data_name] is not None:
            if (self._reader.projection_type == 'Scene') and (scene_args['interval'] == 'auto') and (scene_args['fit_img_size'] is True):
                logging.debug('Use cache: {0}'.format(data_name))
                return self.geo_data_cache[data_name].copy()

            if self._reader.projection_type == 'Tile':
                logging.debug('Use cache: {0}'.format(data_name))
                return self.geo_data_cache[data_name].copy()

        # Get raw data
        req_data = None
        if self._reader.projection_type == 'Scene':
            data = self._reader.get_geometry_data(data_name, interval=scene_args['interval'], fit_img_size=scene_args['fit_img_size'])

            # Set the data into cache space
            (data_size_lin, data_size_pxl) = data.shape
            if (scene_args['fit_img_size'] is True) and (self._reader.img_n_lin <= data_size_lin) and (self._reader.img_n_pix <= data_size_pxl):
                self.geo_data_cache[data_name] = data

            req_data = data.copy()

        elif self._reader.projection_type == 'Tile':
            data = self._reader.get_geometry_data(data_name)

            # Set the data into cache space
            self.geo_data_cache[data_name] = data

            req_data = data.copy()

        return req_data

    def get_product_list(self):
        ret = list(self.h5_file['/Image_data'].keys())
        if self.product_id == 'NWLR':
            ret = ret + ['Rrs_380', 'Rrs_412', 'Rrs_443', 'Rrs_490', 'Rrs_530', 'Rrs_565', 'Rrs_670']

        return ret

    def get_geometry_data_list(self):
        return self._reader.get_geometry_data_list()

    # -----------------------
    # Search methods
    # -----------------------
    def seatch_pos_from_latlon(self, tgt_lat:float, tgt_lon:float):
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
    # File utility
    # -----------------------
    def close(self):
        if self.h5_file is not None:
            logging.debug('Close: {0}'.format(self.filepath))
            self.h5_file.close()

    # +++++++++++++++++++++++
    # Private methods
    # +++++++++++++++++++++++


# =============================
#  Exception Class
# =============================


class NotSupportedError(Exception):
    pass
