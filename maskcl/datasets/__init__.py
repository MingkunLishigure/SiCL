from __future__ import absolute_import
import warnings

from .dukemtmc import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .personx import PersonX
from .veri import VeRi
from .vehicleid import VehicleID
from .vehiclex import VehicleX
from .PRCC import PRCCdataset
from .LTCC import LTCCdataset
from .PRCCMASK import Mask_PRCCdataset
from .LTCCMASK import LTCCMASKdataset
from .nkuhp import NKUHPdataset
from .nkuhpmask import NKUHPMASKdataset
from .Celeblight import CelebRightdataset
from .vc import VCdataset
from .Celeb import Celebdataset
from .real28 import Realdataset


__factory = {
    'market1501': Market1501,
    'dukemtmcreid': DukeMTMCreID,
    'msmt17': MSMT17,
    'personx': PersonX,
    'veri': VeRi,
    'vehicleid': VehicleID,
    'vehiclex': VehicleX,
    'prcc': PRCCdataset,
    'ltcc': LTCCdataset,
    'nkuhp': NKUHPdataset,
    'prccmask': Mask_PRCCdataset,
    'ltccmask': LTCCMASKdataset,
    'nkuhpmask': NKUHPMASKdataset,
    'celebreidlight': CelebRightdataset,
    'VC-Clothes': VCdataset,
    'celebreid': Celebdataset,
    'Real28': Realdataset
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. 
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)
