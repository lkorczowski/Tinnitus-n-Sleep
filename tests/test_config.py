import pytest
import os

def test_Config():
    from tinnsleep.config import Config
    assert os.path.split(Config.bruxisme_files[0])[1] == '1DA15_nuit_hab.edf'