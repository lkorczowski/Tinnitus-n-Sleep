import pytest
import os
import getpass


@pytest.mark.skipif(getpass.getuser()=='runner', reason="Not expecting to work in CI because data required locally")
def test_Config():
    from tinnsleep.config import Config
    assert os.path.split(Config.bruxisme_files[0])[1] == '1DA15_nuit_hab.edf'