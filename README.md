# SPOT: SGLI Python Open Tool

SPOT provides the reading interface, graphic output interface and 
map-projection function of
[GCOM-C/SGLI](https://suzaku.eorc.jaxa.jp/GCOM_C/index.html) 
level-1 and level-2 product data on Python 3.
* Latest Version (Master branch): alpha 1.0

---

## Requirements
* [Python](https://www.python.org/) >= 3.6
* [H5py](https://www.h5py.org/) >= 2.9.0
* [Numpy](https://www.numpy.org/) >= 1.15
* [Matplotlib](https://matplotlib.org/) >= 3.0.3

***

## Usage

**Quick Visualization**
```python
import spot

iwpr_data = spot.File('GC1SG1_YYYYMMDDhhmmsPPPSS_L2SG_IWPRK_vvvvv.h5')
chla_data = iwpr_data.get_product_data('CHLA')
spot.saveimg(iwpr_data, outpath='hoge.png')
```

For more detail, please see [the wiki](https://github.com/K0gata/SGLI_Python_Open_Tool/wiki).

## Installation
Installation on the current version provides manually only. 


1. Download  
    * Git: `$ git clone git@github.com:K0gata/SGLI_Python_Opne_Tool.git`  
    * Web: [Download (zip file)](https://github.com/K0gata/SGLI_Python_Open_Tool/archive/master.zip)
2. Import the SPOT module on your script

```python
import sys
sys.path.append('Directory path for the SPOT')
import spot
```

The PyPI and Conda packages are being prepared.
 
 ## Licence
This project is licensed under the MIT License - see the LICENSE.md file for details.

## SPOT Implementation Status
for **[JAXA Standard Products](https://suzaku.eorc.jaxa.jp/GCOM_C/data/product_std.html)** (HDF5 format):
* Level-1B
  
|  Product | Status   | Remark |
|:--------:|:--------:|:---------:|
|  L1B VNR | Completed | - |
|  L1B IRS | Working   | - |
|  L1B POL | Waiting   | - |


* Level-2 Ocean
  
|  Product | Status   | Remark |
|:--------:|:--------:|:---------:|
|  SST  | Completed  | - |
|  NWLR | Completed  | - |
|  IWPR | Completed  | - |

* Level-2 Terrestrial
  
|  Product | Status   | Remark |
|:--------:|:--------:|:---------:|
|  LTOA | Testing  | - |
|  RSRF | Testing  | - |
|  VGI_ | Testing  | - |
|  LAI_ | Testing  | - |
|  AGB_ | Testing  | - |
|  LST_ | Testing  | - |

* Level-2 Atmosphere
  
|  Product | Status   | Remark |
|:--------:|:--------:|:---------:|
|  CLFG | Working  | - |
|  CLPR | Working  | - |
|  ARNP | Testing  | - |
|  ARPL | Testing  | - |

* Level-2 Cryospher

|  Product | Status   | Remark |
|:--------:|:--------:|:---------:|
|  OKID | Working  | Not supported |
|  SICE | Working  | Not supported |
|  SIPR | Testing  | - |

* Level-2 Global  
Waiting (Not supported)

* Level-3  
*Waiting (Not supported)

for **[JAXA/EORC JASMES Semi-Realtime Data](https://www.eorc.jaxa.jp/cgi-bin/jasmes/sgli_nrt/index.cgi)** (NetCDF4 format):  
Waiting (Not supported)

## Future Update Plan
* File search function
* File download function from [JAXA G-Portal](https://gportal.jaxa.jp/gpr/)
* Mosaic function
* Level-3 interface
* [JASMES semi-realtime data](https://www.eorc.jaxa.jp/cgi-bin/jasmes/sgli_nrt/index.cgi) 
(NetCDF format) interface
* To speed up map-projecting

## Author
[K. Ogata](https://github.com/K0gata)



