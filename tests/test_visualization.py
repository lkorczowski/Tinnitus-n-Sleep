import pytest
import numpy as np
from tinnsleep.visualization import (plotTimeSeries, assert_ax_equals_data,
                                     assert_x_labels_correct, zoom_effect,
                                     plotAnnotations)
import matplotlib.pyplot as plt
import numpy.testing as npt


def test_asserts_homemade():
    """Check if homemade asserts catch the error"""
    sfreq = 1
    np.random.seed(42)
    data = np.random.randn(400, 2)
    fig, ax = plotTimeSeries(data)

    # check if x-labels are incorrect values
    with pytest.raises(AssertionError, match=r"Arrays are not equal\n\nMismatched elements: 399 / 400 (99.8%)*"):
        # check if correct values
        assert_ax_equals_data(data, ax, sfreq=2)

    # check if y-label values are incorrect
    with pytest.raises(AssertionError, match="labels are not the same"):
        assert_x_labels_correct(data, ["0" for k in range(data.shape[1])])

    np.random.seed(10)
    data = np.random.randn(400, 2)

    # check if y-label position are incorrect
    with pytest.raises(AssertionError, match="Arrays are not equal"):
        assert_x_labels_correct(data, ["0" for k in range(data.shape[1])])

    # check if data values are incorrect values
    with pytest.raises(AssertionError, match=r"Items are not equal to 7 significant digits:"):
        # check if correct values
        assert_ax_equals_data(data, ax, sfreq=sfreq)


def test_plotTimeSeries_noparams():
    """Complete test suite for plotTimeSeries
    """
    plt.close("all")
    sfreq = 1
    np.random.seed(42)
    data = np.random.randn(400, 2)
    fig, ax = plotTimeSeries(data)

    # check if label position and values are correct
    assert_x_labels_correct(data, [str(k) for k in range(data.shape[1])])

    # check if correct values
    assert_ax_equals_data(data, ax, sfreq=sfreq)


def test_plotTimeSeries_superimpose():
    """Test if we can superimpose several timeseries
    """
    plt.close("all")
    np.random.seed(42)
    data = np.random.randn(400, 2)
    fig, ax = plotTimeSeries(data, color="red")

    np.random.seed(42)
    data = np.random.randn(400, 4)
    fig, ax = plotTimeSeries(data, ax=ax, color="black", linestyle="--")


def test_plotTimeSeries_chnames_propagation():
    """test if ch_names propagate to all channels
    """
    plt.close("all")
    sfreq = 1
    np.random.seed(42)
    data = np.random.randn(400, 2)
    fig, ax = plotTimeSeries(data, ch_names="EMG")
    # check if label position and values are correct
    assert_x_labels_correct(data, ['EMG' for k in range(data.shape[1])])


def test_plotTimeSeries_subplots():
    """Test if two axes can be managed
    """
    plt.close("all")
    sfreq=100
    np.random.seed(42)
    data = np.random.randn(400, 2)
    ax = plt.subplot(2, 1, 2)
    ch_names=["You underestimate", "my power"]
    fig, ax = plotTimeSeries(data, ax=ax, color="r", marker=".", linestyle='dashed',
                             linewidth=2, markersize=0.5, ch_names=ch_names, sfreq=sfreq)
    plt.title("lava platform")
    assert_ax_equals_data(data, ax, sfreq=sfreq)
    assert_x_labels_correct(data, ch_names)

    sfreq=200
    np.random.seed(42)
    data = np.random.randn(400, 4)
    ax = plt.subplot(2, 1, 1)
    ch_names = ["Its over", "I have the", "high", "ground"]
    fig, ax = plotTimeSeries(data, ax=ax, color="b", marker="*", linestyle='-',
                             linewidth=2, markersize=0.5, ch_names=ch_names, sfreq=sfreq)
    plt.title("higher ground")
    assert_ax_equals_data(data, ax, sfreq=sfreq)
    assert_x_labels_correct(data, ch_names)


def test_plotTimeSeries_1dim():
    np.random.seed(42)
    data = np.random.randn(10)
    plotTimeSeries(data)


def test_plotTimeSeries_incorrectdim():
    np.random.seed(42)
    data = np.random.randn(1, 2, 3)
    with pytest.raises(ValueError, match="data should be two-dimensional"):
        plotTimeSeries(data)


def test_plotTimeSeries_incorrect_parameters():
    np.random.seed(42)
    data = np.random.randn(400, 4)

    with pytest.raises(ValueError, match="\`ch_names\` must be a list or an iterable of shape \(n_dimension,\) or None"):
        plotTimeSeries(data, ch_names=True)

    with pytest.raises(ValueError, match='ch_names should be same length as the number of channels of data'):
        plotTimeSeries(data, ch_names=[1, 2])

    with pytest.raises(ValueError, match="\`ax\` must be a matplotlib Axes instance or None"):
        plotTimeSeries(data, ax=True)


def test_zoom_effect():
    """Test the box connector that allows (in the future) to select specific range of value to show dynamically in
    a jupyter notebook widget
    """
    plt.close("all")
    sfreq=100
    np.random.seed(42)
    data = np.random.randn(400, 2)

    plt.figure(figsize=(5, 5))
    ax1 = plt.subplot(221)
    plotTimeSeries(data, ax=ax1, sfreq=100)
    #ax1.set_xlim(0, 1)
    ax2 = plt.subplot(212)
    plotTimeSeries(data, ax=ax2, sfreq=100)
    zoom_effect(ax1, ax2, 0.2, 0.6)
    ax1.set_xlim(0, 1)

    ax3 = plt.subplot(222)
    plotTimeSeries(data, ax=ax3, sfreq=100)
    zoom_effect(ax3, ax2, fc="red", alpha=0.1, ec="red")
    ax3.set_xlim(2, 2.5)  # move the cursor only with that


def test_zoom_effect2():
    """Test the box connector that allows (in the future) to select specific range of value to show dynamically in
    a jupyter notebook widget
    """
    plt.close("all")
    sfreq=100
    np.random.seed(42)
    data = np.random.randn(400, 2)

    plt.figure(figsize=(5, 5))
    ax1 = plt.subplot(211)
    plotTimeSeries(data, ax=ax1, sfreq=100)
    #ax1.set_xlim(0, 1)
    ax2 = plt.subplot(212)
    plotTimeSeries(data, ax=ax2, sfreq=100)
    zoom_effect(ax1, ax2, prop_lines=dict(linestyle="-."))
    ax1.set_xlim(0.5, 1)


def test_zoom_effect_incorrectparams():
    """Test the box connector that allows (in the future) to select specific range of value to show dynamically in
    a jupyter notebook widget
    """
    plt.close("all")
    sfreq=100
    np.random.seed(42)
    data = np.random.randn(400, 2)

    plt.figure(figsize=(5, 5))
    ax1 = plt.subplot(221)
    plotTimeSeries(data, ax=ax1, sfreq=100)
    #ax1.set_xlim(0, 1)
    ax2 = plt.subplot(212)
    plotTimeSeries(data, ax=ax2, sfreq=100)
    with pytest.raises(ValueError, match=r"xmin & xman should be None or float"):
        zoom_effect(ax1, ax2, xmin=0.2)



def test_plotAnnotations_init():
    plt.close("all")
    annotations = [{'onset': 0.5, 'duration': 1.0, 'description': "blink", 'origin_time': 0.0}]
    plotAnnotations(annotations)


def test_plotAnnotations():
    plt.close("all")
    sfreq=100
    np.random.seed(42)
    data = np.random.randn(400, 2)
    annotations = [{'onset': 0.5, 'duration': 1.0, 'description': "blink", 'origin_time': 0.0}]
    plt.figure(figsize=(5, 5))
    ax1 = plt.subplot(212)
    plotTimeSeries(data, ax=ax1, sfreq=100)
    ax1.set_xlim(0, 2)
    bbox_patches = plotAnnotations(annotations, ax=ax1)

    plt.show()


def test_plotAnnotations2():
    plt.close("all")
    sfreq=100
    np.random.seed(42)
    data = np.random.randn(400, 2)
    annotations = [{'onset': 0.5, 'duration': 1.0, 'description': "blink", 'origin_time': 0.0},
                   {'onset': 1., 'duration': 1.0, 'description': "blink", 'origin_time': 0.0},
                   {'onset': 4.0, 'duration': 10.0, 'description': "dead", 'origin_time': 0.0}
                   ]
    plt.figure(figsize=(5, 5))
    ax1 = plt.subplot(212)
    plotTimeSeries(data, ax=ax1, sfreq=100)
    bbox_patches = plotAnnotations(annotations, ax=ax1)
    ax1.set_xlim(0, 10)


def test_plotAnnotations_incorrectparams():
    plt.close("all")

    annotations = [{'lol': 0.5, 'duration': 1.0, 'description': "blink", 'origin_time': 0.0}]
    with pytest.raises(ValueError, match="lol is an invalid key as annotation"):
        plotAnnotations(annotations)

    annotations = {'lol': 0.5, 'duration': 1.0, 'description': "blink", 'origin_time': 0.0}
    with pytest.raises(ValueError, match="annotations should be a list or ndarray of dict"):
        plotAnnotations(annotations)

    annotations = [{'onset': 0.5, 'duration': 1.0, 'description': "blink", 'origin_time': 0.0}]
    with pytest.raises(ValueError, match=r"`ax` must be a matplotlib Axes instance or None"):
        plotAnnotations(annotations, ax='lol')

    annotations = ["lol", "lol"]
    with pytest.raises(ValueError, match="annotations should contains dict"):
        plotAnnotations(annotations)

