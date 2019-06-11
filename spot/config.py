import warnings
import datetime
import csv
import numpy as np
import os
from spot.projection import PROJ_TYPE

VERSION = 'alpha 0.1'

TZ_UTC0 = datetime.timezone.utc

class SpotWarnings(UserWarning):
    pass

# EOF
