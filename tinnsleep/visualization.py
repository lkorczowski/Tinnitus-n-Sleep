import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
from matplotlib.transforms import (
    Bbox, TransformedBbox, blended_transform_factory)
from mpl_toolkits.axes_grid1.inset_locator import (
    BboxPatch, BboxConnector, BboxConnectorPatch)
import mne

def plotTimeSeries(data,
                   ch_names=None,
                   sfreq=1,
                   scalings=None,
                   ax=None,
                   **kwargs):
    """Advanced plotting of multidimensional time series from ndarray

    Parameters
    ----------
    data: array-line, shape (n_samples, n_dimension)
        multidimensional time series
    ch_names: list | iterable, shape (n_dimension,) | None
        the labels for the time series
    sfreq: float (default: 1)
        sample rate (in Hz)
    scalings: float | iterable, shape (n_dimension,)
        value between two channels
    ax: a instance of ``matplotlib.pyplot.Axes`` (default: None)
        the axe where to save the fig. By default a new figure is generated.

    Returns
    -------
    n_epochs: int
        estimated number of epochs
    """
    shapeD = data.shape
    if len(shapeD) == 1:
        n_channels = 1
        n_samples = shapeD[0]
        data = np.expand_dims(data, axis=1)
    elif len(shapeD) == 2:
        n_channels = shapeD[1]
        n_samples = shapeD[0]
    elif len(shapeD) > 2:
        raise ValueError("data should be two-dimensional")

    if ch_names is None:
        ch_names = np.arange(0, n_channels)
    elif isinstance(ch_names, str):
        ch_names = [ch_names] * n_channels
    elif isinstance(ch_names, (np.ndarray, list)):
        if not len(ch_names) == n_channels:
            raise ValueError('ch_names should be same length as the number of channels of data')
    else:
        msg = "`ch_names` must be a list or an iterable of shape (n_dimension,) or None"
        raise ValueError(msg)

    if ax is None:
        ax = plt.gca()
        fig = ax.figure
    elif isinstance(ax, plt.Axes):
        fig = ax.figure
    else:
        msg = "`ax` must be a matplotlib Axes instance or None"
        raise ValueError(msg)

    # remove median
    data = data - np.median(data, axis=0)

    if scalings is None:
        # distance between two lines: maximum of the 95% percentile of each channel
        scalings = np.max(np.quantile(np.abs(data), 0.975, axis=0))

    # calculate multidimensional shifts based on scalings
    shifts = np.linspace(0, 2 * scalings * (n_channels-1), n_channels)

    # align timeseries with new offsets
    data = data - shifts
    times = np.linspace(0, (n_samples-1) / sfreq, num=n_samples)

    # compute shift based on scalings
    ax.plot(times, data, **kwargs)
    plt.yticks(-shifts, ch_names)
    plt.xlim(np.min(times), np.max(times))
    plt.ylim(np.min(-shifts)-(1.5*scalings), 1.5*scalings)
    return fig, ax



def assert_ax_equals_data(data, ax, sfreq=1):
    """Return assert error if the ax is not comming from data

     Parameters
     ----------
     data: array-line, shape (n_samples, n_dimension)
         multidimensional time series
     ax: a instance of ``matplotlib.pyplot.Axes``
        the ax where the data were plotted
     sfreq: float (default: 1)
         sample rate (in Hz)
     """
    # check if correct values
    for n, line in enumerate(ax.get_lines()):
        x, y = line.get_data()
        # check if data correlated perfectly (there aren't equal due to transformation)
        npt.assert_approx_equal(np.corrcoef(y, data[:, n])[0][1], 1)  # data y-axis correlate to 1
        npt.assert_equal(x, np.linspace(0, (data.shape[0]-1)/sfreq, data.shape[0]))  # time x-axis match


def assert_x_labels_correct(data, expected_labels):
    """Return assert error if the ax is not comming from data

     Parameters
     ----------
     data: array-line, shape (n_samples, n_dimension)
         multidimensional time series
     expected_labels: a list of str
         the expected label names
     """
    # prepare data to double check
    data = data - np.median(data, axis=0)
    scalings = np.max(np.quantile(np.abs(data), 0.975, axis=0))

    # calculate multidimensional shifts based on scalings
    shifts = - np.linspace(0, 2 * scalings * (data.shape[1]-1), data.shape[1])

    # check if label position and values are correct
    locs, labels = plt.yticks()
    npt.assert_equal(locs, shifts)
    for k, label in enumerate(labels):
        assert label.get_text() == expected_labels[k], "labels are not the same"


def plotAnnotations(annotations, ax=None, text_prop={}, **kwargs):
    """Add a box for each annotation

    Parameters
    ----------
    annotations: a instance mne.Annotations | list of dictionary | dict
        a list of annotation or dictionary containing the following fields:
        {'onset': float (seconds), 'duration': float (seconds), 'description': str, orig_time': float (seconds)}
        Example:
        >>> # a list of two annotations starting after 0.5 and 1 second of duration 1.0 second named 'blink'
        >>> annotations = [{'onset': 0.5, 'duration': 1.0, 'description': "blink", 'origin_time': 0.0},
        >>>                {'onset': 1., 'duration': 1.0, 'description': "blink", 'origin_time': 0.0}]
        or
        >>> # a list of two annotations starting after 0.5 and 1 second of duration 1.0 second named 'blink'
        >>> annotations = {'onset': [0.5, 1.0], 'duration': [1.0, 1.0],
        >>>                'description': ["blink", "blink"], 'origin_time': [0., 0.]}
    ax: a instance of ``matplotlib.pyplot.Axes`` (default: None)
        the axe where to save the fig. By default a new figure is generated.
    text_prop: dict
        parameters send to ``matplotlib.text.Text`` instance
    **kwargs
        Arguments passed to the patch constructor `mpl_toolkits.axes_grid1.inset_locator.BboxPatch`` (e.g. fc, ec).

    """
    if isinstance(annotations, (np.ndarray, list)):
        for annotation in annotations:
            if not isinstance(annotation, dict):
                raise ValueError("annotations should contains dict")
            else:
                for key in annotation.keys():
                    if key not in ['onset', 'duration', 'description', 'origin_time']:
                        raise ValueError(f"{key} is an invalid key as annotation")
    elif isinstance(annotations, mne.Annotations):
        pass
    else:
        raise ValueError("annotations should be a list or ndarray of dict")

    if ax is None:
        ax = plt.gca()
        fig = ax.figure
    elif isinstance(ax, plt.Axes):
        fig = ax.figure
    else:
        msg = "`ax` must be a matplotlib Axes instance or None"
        raise ValueError(msg)

    if text_prop == {}:
        text_prop = {"color": "red"}
    if kwargs == {}:
        kwargs = {**text_prop, "ec": "none", "alpha": 0.2}

    for annotation in annotations:
        xmin = annotation["onset"]
        xmax = xmin + annotation["duration"]
        trans = blended_transform_factory(ax.transData, ax.transAxes)

        bbox = Bbox.from_extents(xmin, 0, xmax, 1)
        mybbox = TransformedBbox(bbox, trans)

        bbox_patch = BboxPatch(mybbox, **kwargs)
        ax.add_patch(bbox_patch)
        ax.text(np.mean([xmin, xmax]), 1.05, annotation["description"], transform=trans,
                horizontalalignment='center',
                **text_prop)


def connect_bbox(bbox1, bbox2,
                 loc1a, loc2a, loc1b, loc2b,
                 prop_lines, **prop_patches):
    """ Create patch and lines to connect two bbox instances' opposite corner together

    Parameters
    ----------
    bbox1, bbox2 : `matplotlib.transforms.Bbox`
        Bounding boxes to connect.

    loc1a, loc2a : {1, 2, 3, 4}
        Corners of *bbox1* and *bbox2* to draw the first line.
        Valid values are::

            'upper right'  : 1,
            'upper left'   : 2,
            'lower left'   : 3,
            'lower right'  : 4

    loc1b, loc2b : {1, 2, 3, 4}
        Corners of *bbox1* and *bbox2* to draw the second line.
        Valid values are::

            'upper right'  : 1,
            'upper left'   : 2,
            'lower left'   : 3,
            'lower right'  : 4

    propo_lines:
            Patch properties for the line drawn:
            %(Patch)s

    Returns
    -------
    c1 : a instance of mpl_toolkits.axes_grid1.inset_locator.BboxConnector
    c2 : a instance of mpl_toolkits.axes_grid1.inset_locator.BboxConnector
    bbox_patch1 : a instance of mpl_toolkits.axes_grid1.inset_locator.BboxPatch
    bbox_patch2 : a instance of mpl_toolkits.axes_grid1.inset_locator.BboxPatch
    p : a instance of mpl_toolkits.axes_grid1.inset_locator.BboxConnectorPatch

    Reference
    ---------
    https://matplotlib.org/3.1.1/gallery/subplots_axes_and_figures/axes_zoom_effect.html
    """
    if prop_patches == {}:
        prop_patches = {
            **prop_lines,
            "alpha": prop_lines.get("alpha", 1) * 0.2,
        }

    # build two lines
    c1 = BboxConnector(bbox1, bbox2, loc1=loc1a, loc2=loc2a, **prop_lines)
    c1.set_clip_on(False)
    c2 = BboxConnector(bbox1, bbox2, loc1=loc1b, loc2=loc2b, **prop_lines)
    c2.set_clip_on(False)

    #
    bbox_patch1 = BboxPatch(bbox1, **prop_patches)
    bbox_patch2 = BboxPatch(bbox2, **prop_patches)

    p = BboxConnectorPatch(bbox1, bbox2,
                           loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
                           **prop_lines
                           )
    p.set_clip_on(False)

    return c1, c2, bbox_patch1, bbox_patch2, p


def zoom_effect(ax1, ax2, xmin=None, xmax=None, prop_lines={}, **kwargs):
    """
    Connect *ax1* and *ax2*. The *xmin*-to-*xmax* range in both axes will be marked.

    Parameters
    ----------
    ax1
        The zoomed axes.
    ax2
        The main axes.
    xmin, xmax
        The limits of the colored area in both plot axes. If None, xmin & xmax will be taken from the ax1.viewLim.
    prop_lines: dict (default: {})
        Arguments passed to the line constructor ``mpl_toolkits.axes_grid1.inset_locator.BboxConnector``
    **kwargs
        Arguments passed to the patch constructor `mpl_toolkits.axes_grid1.inset_locator.BboxPatch`` (e.g. fc, ec).

    References
    ----------
    https://matplotlib.org/3.1.1/gallery/subplots_axes_and_figures/axes`_zoom_effect.html
    """

    # with auto-xlim based on the x2 xlim
    if (xmin is None) and (xmax is None):
        tt = ax1.transScale + (ax1.transLimits + ax2.transAxes)
        trans = blended_transform_factory(ax2.transData, tt)
        mybbox1 = ax1.bbox
        mybbox2 = TransformedBbox(ax1.viewLim, trans)

    # with specific xlim
    elif isinstance(xmin, float) and isinstance(xmax, float):
        trans1 = blended_transform_factory(ax1.transData, ax1.transAxes)
        trans2 = blended_transform_factory(ax2.transData, ax2.transAxes)
        bbox = Bbox.from_extents(xmin, 0, xmax, 1)
        mybbox1 = TransformedBbox(bbox, trans1)
        mybbox2 = TransformedBbox(bbox, trans2)
    else:
        raise ValueError("xmin & xman should be None or float")

    c1, c2, bbox_patch1, bbox_patch2, p = connect_bbox(mybbox1, mybbox2,
                                                       loc1a=3, loc2a=2, loc1b=4, loc2b=1,
                                                       prop_lines=prop_lines, **kwargs)
    ax1.add_patch(bbox_patch1)
    ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    p.set_clip_on(False)
    ax2.add_patch(p)

    return c1, c2, bbox_patch1, bbox_patch2, p

