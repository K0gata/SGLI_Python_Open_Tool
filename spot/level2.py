import numpy as np
import logging
from decimal import Decimal, ROUND_HALF_UP
from abc import ABC, abstractmethod, abstractproperty
from spot.utility import bilin_2d
from spot.config import PROJ_TYPE

# =============================
#  Level-2 template class
# =============================
class L2Interface(ABC):
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

    def get_geometry_data(self, data_name:str, **kwargs):
        interval = kwargs['interval']

        dset = self.h5_file['Geometry_data/' + data_name]
        data = dset[:]
        if 'Latitude' is not data_name and 'Longitude' is not data_name:
            data = dset[:].astype(np.float32)
            if 'Error_DN' in dset.attrs:
                data[data == dset.attrs['Error_DN'][0]] = np.NaN
            with np.warnings.catch_warnings():
                np.warnings.filterwarnings('ignore', r'invalid value encountered in (greater|less)')
                if 'Maximum_valid_DN' in dset.attrs:
                    data[data > dset.attrs['Maximum_valid_DN'][0]] = np.NaN
                if 'Minimum_valid_DN' in dset.attrs:
                    data[data < dset.attrs['Minimum_valid_DN'][0]] = np.NaN

            data = data * dset.attrs['Slope'][0] + dset.attrs['Offset'][0]

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

    @abstractmethod
    def get_geometry_data_list(self):
        raise NotImplementedError()

    def get_product_data_list(self):
        return list(self.h5_file['/Image_data'].keys())

    def get_unit(self, prod_name: str):
        # Get attrs set
        unit_name = 'Unit'
        if prod_name in self.get_product_data_list():
            grp_name = '/Image_data/'
        else:
            grp_name = '/Geometry_data/'
        attrs = self.h5_file[grp_name + prod_name].attrs

        # Get unit
        if unit_name not in attrs:
            return 'NA'
        return attrs[unit_name][0].decode('UTF-8')

# =============================
# Level-2 map-projection class
# =============================


class Tile(L2Interface):
    PROJECTION_TYPE = PROJ_TYPE.TILE.name
    ALLOW_PROJECTION_TYPE = [PROJECTION_TYPE, PROJ_TYPE.EQR.name]

    def __init__(self, h5_file, product_id):
        super().__init__(h5_file, product_id)

        glb_attrs = h5_file['/Global_attributes'].attrs
        if 'Tile_number' in glb_attrs:
            tile_numbner = glb_attrs['Tile_number'][0].decode('UTF-8')
            self.vtile = int(tile_numbner[0:2])
            self.htile = int(tile_numbner[2:])
        else:
            product_file_name = glb_attrs['Product_file_name'][0].decode('UTF-8')
            tile_numbner = product_file_name.split('_')[2]
            self.vtile = int(tile_numbner[1:3])
            self.htile = int(tile_numbner[3:])

        self.img_spatial_reso = h5_file['/Image_data'].attrs['Grid_interval'][0]
        if self.img_spatial_reso > 0.0083 and self.img_spatial_reso < 0.0084:
            self.img_spatial_reso = 1000.
        elif self.img_spatial_reso > 0.002 and self.img_spatial_reso < 0.003:
            self.img_spatial_reso = 250.

    def get_geometry_data(self, data_name:str, **kwargs):

        if data_name == 'Latitude' or data_name == 'Longitude':
            # This algorithm referred to 'GCOM-C Data Users Handbook (SGC-180024)'
            lin_num = np.tile(np.arange(0, self.img_n_lin).reshape(-1,1), (1, self.img_n_pix))
            col_num = np.tile(np.arange(0, self.img_n_pix), (self.img_n_lin, 1))

            d = 180. / self.img_n_lin / 18
            np_0 = 2 * float(Decimal(180. / d).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
            lin_total = lin_num + (self.vtile * self.img_n_lin)
            col_total = col_num + (self.htile * self.img_n_pix)

            lat = 90. - (lin_total + 0.5) * d
            if data_name == 'Latitude':
                return lat
            elif data_name == 'Longitude':
                np_i = (np_0 * np.cos(np.deg2rad(lat)) + 0.5).astype(np.int32).astype(np.float32)
                lon = 360. / np_i * (col_total - np_0 / 2. + 0.5)

                if self.htile > 17:
                    lon[lon > 180] = np.NaN
                else:
                    lon[lon < -180] = np.NaN
                return lon
        else:
            return super().get_geometry_data(data_name, **kwargs)

        return None

    def get_geometry_data_list(self):
        return ['Latitude', 'Longitude'] + list(self.h5_file['/Geometry_data'].keys())

    def get_allow_projection_type(self):
        return self.ALLOW_PROJECTION_TYPE

    def get_unit(self, prod_name: str):
        if prod_name == 'Latitude' or prod_name == 'Longitude':
            return 'degree'

        return super().get_unit(prod_name)

class Scene(L2Interface):
    PROJECTION_TYPE = PROJ_TYPE.SCENE.name
    ALLOW_PROJECTION_TYPE = [PROJECTION_TYPE, PROJ_TYPE.EQR.name]

    def __init__(self, h5_file, product_id):
        super().__init__(h5_file, product_id)
        self.scene_number = h5_file['/Global_attributes'].attrs['Scene_number'][0]
        self.path_number = h5_file['/Global_attributes'].attrs['RSP_path_number'][0]

        img_data_grp_attrs = self.h5_file['Image_data'].attrs
        self.img_spatial_reso = img_data_grp_attrs['Grid_interval'][0]

    def get_geometry_data_list(self):
        return list(self.h5_file['/Geometry_data'].keys())

    def get_allow_projection_type(self):
        return self.ALLOW_PROJECTION_TYPE

# =============================
# Level-2 area class
# =============================


class RadianceL2(Tile):
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


    def get_product_data_list(self):
        prod_list = super().get_product_data_list()
        for prod in prod_list:
            if ('Lt_P' in prod) or ('Lt_S' in prod) or ('Lt_V' in prod):
                prod_list.append(prod.replace('Lt', 'Rt'))
                prod_list.append(prod.replace('Lt', 'Stray_light_correction_flag'))

        prod_list = sorted(prod_list)
        return prod_list

    def get_unit(self, prod_name: str):
        if 'Rt_' in prod_name:
            return 'NA'

        return super().get_unit(prod_name)

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

        solz_name = 'Solar_zenith'
        if 'Lt_P' in prod_name:
            solz_name = 'Solar_zenith_PL'
        cos_theta_0 = np.cos(np.deg2rad(self.get_geometry_data(solz_name, interval='auto', fit_img_size=True)))
        data = data / cos_theta_0

        return data

    def _get_stray_light_correction_flag(self, prod_name):
        prod_name = prod_name.replace('Stray_light_correction_flag_', 'Lt_')
        dset = self.h5_file['Image_data/' + prod_name]
        dn_data = dset[:]
        data = np.bitwise_and(dn_data, 0x8000)
        data[dn_data == dset.attrs['Error_DN']] = 0

        return data > 0

class OceanL2(Scene):
    def get_product_data(self, prod_name:str):
        if 'Rrs' not in prod_name:
            return super().get_product_data(prod_name)

        # Get Rrs data
        real_prod_name = prod_name.replace('Rrs', 'NWLR')
        dset = self.h5_file['Image_data/' + real_prod_name]

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
        data = data * dset.attrs['Rrs_slope'][0] + dset.attrs['Rrs_offset'][0]

        return data

    def get_product_data_list(self):
        prod_list = super().get_product_data_list()
        if self.product_id == 'NWLR':
            prod_list = prod_list + ['Rrs_380', 'Rrs_412', 'Rrs_443', 'Rrs_490', 'Rrs_530', 'Rrs_565', 'Rrs_670']

        return prod_list

    def get_unit(self, prod_name: str):
        # Get attrs set
        unit_name = 'Unit'
        real_prod_name = prod_name
        if 'Rrs' in prod_name:
            real_prod_name = prod_name.replace('Rrs', 'NWLR')
            unit_name = 'Rrs_unit'
        attrs = self.h5_file['/Image_data/' + real_prod_name].attrs

        # Get unit
        if unit_name not in attrs:
            return 'NA'
        return attrs[unit_name][0].decode('UTF-8')


class LandL2(Tile):

    def get_product_data(self, prod_name: str):
        return super().get_product_data(prod_name)


class AtmosphereL2(Tile):

    def get_product_data(self, prod_name: str):
        return super().get_product_data(prod_name)


class CryosphereL2(Tile):

    def get_product_data(self, prod_name: str):
        return super().get_product_data(prod_name)


class CryosphereOkhotskL2(Scene):

    def get_product_data(self, prod_name: str):
        return super().get_product_data(prod_name)

    def get_flag(self, *args):
        return None

# EOF
