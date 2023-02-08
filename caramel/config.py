import numpy as np
from dataclasses import dataclass
import datetime

@dataclass
class satellite:
    vcd: np.ndarray
    time: datetime.datetime
    profile: np.ndarray
    tropopause: np.ndarray
    latitude_corner: np.ndarray
    longitude_corner: np.ndarray
    uncertainty: np.ndarray
    quality_flag: np.ndarray
    pressure_mid: np.ndarray
    averaging_kernels: np.ndarray
    amf_total: np.ndarray

@dataclass
class ctm_model:
    latitude: np.ndarray
    longitude: np.ndarray
    gas_profile: dict
    pressure_mid: np.ndarray
    tempeature_mid: np.ndarray
    delta_p: np.ndarray
