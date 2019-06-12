import numpy as np
import logging
from decimal import Decimal, ROUND_HALF_UP
from abc import ABC, abstractmethod, abstractproperty
from spot.utility import bilin_2d
from spot.config import PROJ_TYPE

# =============================
#  Level-1 template class
# =============================


class L1Interface(ABC):
    @property
    @abstractmethod
    def PROJECTION_TYPE(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def ALLOW_PROJECTION_TYPE(self):
        return NotImplementedError()

    def __init__(self, h5_file, product_id):
        self.h5_file = h5_file
        self.product_id = product_id

        geo_data_grp_attrs = self.h5_file['Geometry_data'].attrs
        self.geo_n_pix = geo_data_grp_attrs['Number_of_pixels'][0]
        self.geo_n_lin = geo_data_grp_attrs['Number_of_lines'][0]

        img_data_grp_attrs = self.h5_file['Image_data'].attrs
        self.img_n_pix = img_data_grp_attrs['Number_of_pixels'][0]
        self.img_n_lin = img_data_grp_attrs['Number_of_lines'][0]

    def get_product_data(self, prod_name:str):
        dset = self.h5_file['Image_data/' + prod_name]

        # Return uint16 type data if the product is QA_flag or Line_tai93
        if 'QA_flag' == prod_name or 'Line_tai93' == prod_name:
            return dset[:]

        # Validate
        data = dset[:].astype(np.float32)
        if 'Error_DN' in dset.attrs:
            data[data == dset.attrs['Error_DN'][0]] = np.NaN
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'invalid value encountered in (greater|less)')
            if 'Maximum_valid_DN' in dset.attrs:
                data[data > dset.attrs['Maximum_valid_DN'][0]] = np.NaN
            if 'Minimum_valid_DN' in dset.attrs:
                data[data < dset.attrs['Minimum_valid_DN'][0]] = np.NaN

        # Convert DN to physical value
        data = data * dset.attrs['Slope'][0] + dset.attrs['Offset'][0]

        return data

    @abstractmethod
    def get_geometry_data(self, data_name:str, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_geometry_data_list(self):
        raise NotImplementedError()

    def get_product_data_list(self):
        return list(self.h5_file['/Image_data'].keys())

    def get_unit(self, prod_name: str):
        if 'Rt_' in prod_name:
            return 'NA'

        # Get attrs set
        unit_name = 'Unit'
        attrs = self.h5_file['/Image_data/' + prod_name].attrs

        # Get unit
        if unit_name not in attrs:
            return 'NA'
        return attrs[unit_name][0].decode('UTF-8')

# =============================
# Level-1 map-projection class
# =============================


class Scene(L1Interface):
    PROJECTION_TYPE = PROJ_TYPE.SCENE.name
    ALLOW_PROJECTION_TYPE = [PROJECTION_TYPE, PROJ_TYPE.EQR.name]

    def __init__(self, h5_file, product_id):
        super().__init__(h5_file, product_id)
        self.scene_number = h5_file['/Global_attributes'].attrs['Scene_number'][0]
        self.path_number = h5_file['/Global_attributes'].attrs['RSP_path_number'][0]

        img_data_grp_attrs = self.h5_file['Image_data'].attrs
        self.img_spatial_reso = img_data_grp_attrs['Grid_interval'][0]

    def get_geometry_data(self, data_name: str, **kwargs):
        interval = kwargs['interval']

        dset = self.h5_file['Geometry_data/' + data_name]
        data = dset[:]
        if 'Latitude' is not data_name and 'Longitude' is not data_name:
            data = data.astype(np.float32) * dset.attrs['Slope'][0] + dset.attrs['Offset'][0]

        # Finish if interval is none
        if interval is None or interval == 'none':
            return data

        # Interpolate raw data
        if interval == 'auto':
            interp_interval = dset.attrs['Resampling_interval'][0]
        else:
            interp_interval = interval

        lon_mode = False
        if 'Longitude' == data_name:
            lon_mode = True
        if interp_interval > 1:
            data = bilin_2d(data, interp_interval, lon_mode)

        # Trim away the excess pixel/line
        (data_size_lin, data_size_pxl) = data.shape
        if (kwargs['fit_img_size'] is True) and (self.img_n_lin <= data_size_lin) and (self.img_n_pix <= data_size_pxl):
            data = data[:self.img_n_lin, :self.img_n_pix]

        return data

    def get_geometry_data_list(self):
        return list(self.h5_file['/Geometry_data'].keys())

    def get_allow_projection_type(self):
        return self.ALLOW_PROJECTION_TYPE

# =============================
# Level-1 sub-processing level class
# =============================


class L1B(Scene):

    # -----------------------------
    # Public
    # -----------------------------
    def get_product_data(self, prod_name:str):
        if 'Land_water_flag' in prod_name:
            return self._get_land_water_flag()

        if 'Lt_' in prod_name:
            return self._get_Lt(prod_name)

        if 'Rt_' in prod_name:
            return self._get_Rt(prod_name)

        if 'Stray_light_correction_flag_' in prod_name:
            return self._get_stray_light_correction_flag(prod_name)

        return super().get_product_data(prod_name)

    # -----------------------------
    # Private
    # -----------------------------
    def _get_land_water_flag(self):
        dset = self.h5_file['Image_data/Land_water_flag']
        data = dset[:].astype(np.float32)
        if 'Error_DN' in dset.attrs:
            data[data == dset.attrs['Error_value'][0]] = np.NaN
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'invalid value encountered in (greater|less)')
            data[data > dset.attrs['Maximum_valid_value'][0]] = np.NaN
            data[data < dset.attrs['Minimum_valid_value'][0]] = np.NaN

        return data

    def _get_Lt(self, prod_name):
        dset = self.h5_file['Image_data/' + prod_name]
        dn_data = dset[:]
        mask = dset.attrs['Mask'][0]
        data = np.bitwise_and(dn_data, mask).astype(np.float32)
        data = data * dset.attrs['Slope'] + dset.attrs['Offset']
        data[dn_data == dset.attrs['Error_DN']] = np.NaN
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'invalid value encountered in (greater|less)')
            data[data > dset.attrs['Maximum_valid_DN'][0]] = np.NaN
            data[data < dset.attrs['Minimum_valid_DN'][0]] = np.NaN

        return data

    def _get_Rt(self, prod_name):
        prod_name = prod_name.replace('Rt_', 'Lt_')
        dset = self.h5_file['Image_data/' + prod_name]
        dn_data = dset[:]
        mask = dset.attrs['Mask'][0]
        data = np.bitwise_and(dn_data, mask).astype(np.float32)
        data = data * dset.attrs['Slope_reflectance'] + dset.attrs['Offset_reflectance']
        data[dn_data == dset.attrs['Error_DN']] = np.NaN
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'invalid value encountered in (greater|less)')
            data[data > dset.attrs['Maximum_valid_DN'][0]] = np.NaN
            data[data < dset.attrs['Minimum_valid_DN'][0]] = np.NaN

        cos_theta_0 = np.cos(np.deg2rad(self.get_geometry_data('Solar_zenith', interval='auto', fit_img_size=True)))
        data = data / cos_theta_0

        return data

    def _get_stray_light_correction_flag(self, prod_name):
        prod_name = prod_name.replace('Stray_light_correction_flag_', 'Lt_')
        dset = self.h5_file['Image_data/' + prod_name]
        dn_data = dset[:]
        data = np.bitwise_and(dn_data, 0x8000)
        data[dn_data == dset.attrs['Error_DN']] = 0

        return data > 0


class VNRL1B(L1B):

    def get_product_data_list(self):
        prod_list = super().get_product_data_list()
        for prod in prod_list:
            if 'Lt_' in prod:
                prod_list.append(prod.replace('Lt', 'Rt'))
                prod_list.append(prod.replace('Lt', 'Stray_light_correction_flag'))

        prod_list = sorted(prod_list)
        return prod_list


class IRSL1B(L1B):

    def get_product_data_list(self):
        prod_list = super().get_product_data_list()
        for prod in prod_list:
            if 'Lt_SW' in prod:
                prod_list.append(prod.replace('Lt', 'Rt'))
                prod_list.append(prod.replace('Lt', 'Stray_light_correction_flag'))

        prod_list = sorted(prod_list)
        return prod_list

# EOF
