import pytest
import numpy as np
import numpy.testing as npt
from tinnsleep.data import RawToEpochs_sliding,CreateRaw
from tinnsleep.visualization import create_visual
from tinnsleep.utils import compute_nb_epochs


def test_visualization():
    data = np.random.randn(2, 400)
    ch_names = ['Fz', 'Pz']
    duration = 0.1 #in seconds
    interval =0.1  #in seconds
    raw = CreateRaw(data, ch_names)
    leny=compute_nb_epochs(400, 0.1*200, 0.1*200)  #important : convert the 
    #duration and interval in number of samples
    b_detect=[False for i in range(leny)]
    b_detect[2]=True
    create_visual(raw, b_detect, duration, interval)