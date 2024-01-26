"""
Main init file
"""

import pkg_resources

from .base.apipkg import initpkg

__all__ = ('__version__', )
__version__ = '1.0'

initpkg(
    __name__,
    exportdefs = {
        'Test' : '.test.test:Test_var',
        'SSMLearn' : '.main.main:SSMLearn'
    },
)

import logging

logging.basicConfig(
    format='%(levelname) -6s %(asctime)s %(module)s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)
