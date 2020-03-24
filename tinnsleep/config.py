""" Configuration methods for tinnsleep loading/testing
"""

from glob import glob
import os
import getpass

class Config():
    """Load user-dependent variable and path
    """

    username = getpass.getuser()
    known_users = {"robin": 'C:\\Users\\zeta\\documents\\EEG_polysomno\\PSG_tamtin_nox',
                    "louis": "/Users/louis/Data/SIOPI/bruxisme"}

    if username in known_users.keys():
        data_path = known_users[username]
    else:
        raise KeyError(username + ": User path not defined, please add path in utils.config.")

    bruxisme_files = sorted(glob(os.path.join(data_path, '*edf')))