r"""
stLVG
"""

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from pkg_resources import get_distribution
    version = lambda name: get_distribution(name).version

from . import viz
from . import model

name = "stLVG_upload"
# __version__ = version(name)
__author__ = 'Yikai Lou'