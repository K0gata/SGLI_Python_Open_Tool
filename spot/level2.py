import numpy as np
from PIL import Image
import logging
from decimal import Decimal, ROUND_HALF_UP
from abc import ABC, abstractmethod, abstractproperty

# =============================
#  Level-2 template class
# =============================


class L2Interface(ABC):
    @property
    @abstractmethod
    def projection_type(self):
        raise NotImplementedError()

    def __init__(self, h5_file, product_id):
        self.h5_file = h5_file
        self.product_id = product_id

        geo_data_grp_attrs = self.h5_file['Geometry_data'].attrs
        self.geo_n_pix = geo_data_grp_attrs['Number_of_pixels'][0]
        self.geo_n_lin = geo_data_grp_attrs['Number_of_lines'][0]

        img_data_grp_attrs = self.h5_file['Image_data'].attrs
        self.img_n_pix = img_data_grp_attrs['Number_of_pixels'][0]
        self.img_n_lin = img_data_grp_attrs['Number_of_lines'][0]

    def get_product(self, prod_name:str):
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

    def get_product_list(self):
        return list(self.h5_file['/Image_data'].keys())

# =============================
# Level-2 map-projection class
# =============================


class Tile(L2Interface):
    projection_type = 'Tile'

    def __init__(self, h5_file, product_id):
        super().__init__(h5_file, product_id)
        tile_numbner = h5_file['/Global_attributes'].attrs['Tile_number'][0].decode('UTF-8')
        self.vtile = int(tile_numbner[0:2])
        self.htile = int(tile_numbner[2:])

    def get_geometry_data(self, data_name:str, **kwargs):
        # This algorithm referred to 'GCOM-C Data Users Handbook (SGC-180024)'
        lin_num = np.tile(np.arange(0, self.img_n_lin).reshape(-1,1), (1, self.img_n_pix))
        col_num = np.tile(np.arange(0, self.img_n_pix), (self.img_n_lin, 1))

        d = 180. / self.img_n_lin / 18
        nl = 180. / d
        np_0 = 2 * float(Decimal(180. / d).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
        lin_total = lin_num + (self.vtile * self.img_n_lin)
        col_total = col_num + (self.htile * self.img_n_pix)

        lat = 90. - (lin_total + 0.5) * d
        if data_name == 'Latitude':
            return lat
        elif data_name == 'Longitude':
            np_i = (np_0 * np.cos(np.deg2rad(lat)) + 0.5).astype(np.int32).astype(np.float32)
            lon = 360. / np_i * (col_total - np_0 / 2. + 0.5)
            return lon

        return None

    def get_geometry_data_list(self):
        return ['Latitude', 'Longitude']

class Scene(L2Interface):
    projection_type = 'Scene'

    def __init__(self, h5_file, product_id):
        super().__init__(h5_file, product_id)
        self.scene_number = h5_file['/Global_attributes'].attrs['Scene_number'][0]
        self.path_number = h5_file['/Global_attributes'].attrs['RSP_path_number'][0]

    def get_geometry_data(self, data_name:str, **kwargs):
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
        logging.debug('Interval: {0}'.format(interp_interval))

        if interp_interval > 1:
            pil_data = Image.fromarray(data, 'F')
            pil_data = pil_data.resize((self.geo_n_pix * interp_interval, self.geo_n_lin * interp_interval),
                                       Image.BILINEAR)
            data = np.asarray(pil_data)

        # Trim away the excess pixel/line
        (data_size_lin, data_size_pxl) = data.shape
        if (kwargs['fit_img_size'] is True) and (self.img_n_lin <= data_size_lin) and (self.img_n_pix <= data_size_pxl):
            data = data[:self.img_n_lin, :self.img_n_pix]

        return data

    def get_geometry_data_list(self):
        return list(self.h5_file['/Geometry_data'].keys())

# =============================
# Level-2 area class
# =============================


class OceanL2(Scene):
    def get_product(self, prod_name:str):
        if 'Rrs' not in prod_name:
            return super().get_product(prod_name)

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

    def get_product_list(self):
        prod_list = super().get_product_list()
        if self.product_id == 'NWLR':
            prod_list = prod_list + ['Rrs_380', 'Rrs_412', 'Rrs_443', 'Rrs_490', 'Rrs_530', 'Rrs_565', 'Rrs_670']

        return prod_list

class LandL2(Tile):

    def get_product(self, prod_name: str):
        return super().get_product(prod_name)


class AtmosphereL2(Tile):

    def get_product(self, prod_name: str):
        return super().get_product(prod_name)


class CryosphereL2(Tile):

    def get_product(self, prod_name: str):
        return super().get_product(prod_name)


class CryosphereOkhotskL2(Scene):

    def get_product(self, prod_name: str):
        return super().get_product(prod_name)

    def get_flag(self, *args):
        return None
