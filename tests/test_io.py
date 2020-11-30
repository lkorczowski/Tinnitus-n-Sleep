import mne
from tinnsleep.io import fif2edf
import os
import numpy as np
import numpy.testing as npt
from tinnsleep.external.save_edf import write_edf
import numpy.testing as npt


def test_fif2edf():
    fname = "./dummy_fif"
    fname_edf = fname +".edf"
    raw = mne.io.read_raw_fif(fname+".fif", preload=True)
    raw.plot()
    signals = raw.get_data()
    N, T = signals.shape
    scaling = np.ones((signals.shape[0]))
    scaling[np.median(np.abs(signals), axis=1)< 1e-3] = 1e6
    #raw.info["ch_names"]
    if os.path.exists(fname_edf):
        os.remove(fname_edf)
    write_edf(raw, fname_edf)

    raw2 = mne.io.read_raw_edf(fname_edf)
    signals2 = raw2.get_data()
    is_uV = np.median(np.abs(signals), axis=1)< 1e-3
    npt.assert_allclose(signals2[is_uV,:T], signals[is_uV,:],atol=0.01)
    npt.assert_allclose(signals2[~is_uV,:T], signals[~is_uV,:],atol=0.01)

    #raw2.plot()
