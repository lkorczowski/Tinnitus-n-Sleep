import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
from matplotlib.transforms import (
    Bbox, TransformedBbox, blended_transform_factory)
from mpl_toolkits.axes_grid1.inset_locator import (
    BboxPatch, BboxConnector, BboxConnectorPatch)
import mne
from tinnsleep.validation import is_valid_ch_names
import seaborn as sns
import scipy
import pandas as pd


def plotTimeSeries(data,
                   ch_names=None,
                   sfreq=1,
                   scalings=None,
                   ax=None,
                   offset=0,
                   **kwargs):
    """Advanced plotting of multidimensional time series from numpy ndarray in one single matplotlib ax
    Useful for physiological timeseries such as EEG, EMG, MEG, etc.

    Works better for centered time series (zero-mean) and with same order of magnitude variance. You can try to
    normalize if needed (e.g. subtracting mean and dividing by the variance of each individual channels).

    License: The MIT Licence
    Copyright: Louis Korczowski <louis.korczowski@gmail.com>, 2020.

    Parameters
    ----------
    data: array-line, shape (n_samples, n_dimension)
        multidimensional time series
    ch_names: list | iterable, shape (n_dimension,) | None (default: None)
        the labels for the time series, if None the channels are named by their numerical index.
    sfreq: float (default: 1)
        sample rate (in Hz)
    scalings: float | None (default: None)
        value between two channels, If None, try to find the best scalings automatically.
    ax: a instance of ``matplotlib.pyplot.Axes`` (default: None)
        the axe where to save the fig. By default a new figure is generated.
    offset: float (default:0)
        offset for the xlabels in seconds (or samples if `sfreq=1`)
        e.g.: `offset=-2` the axis will starts at at -2
    **kwargs:
        parameters to pass to plt.plot()

    Returns
    -------
    fig: a `matplotlib.figure.Figure` instance
        the linked figure instance (if `ax=None` then it is a new figure)
    ax: a instance of ``matplotlib.pyplot.Axes`` (default: None)
        the linked axe

    Example
    -------
    The following example will output four channels timeseries into a unique ax using subplot (automatic scalings)

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> data = np.random.randn(200, 4)
    >>> ax = plt.subplot(212)
    >>> plotTimeSeries(data, ax=ax)
    >>> plt.show()

    The following example superpose two two channels timeseries for comparison.
    Note that automatic scalings is robust to artifacts because of the use of median.
    >>> data = np.random.randn(400, 2)
    >>> ax = plt.subplot(212)
    >>> plotTimeSeries(data, ax=ax, color="black")
    >>> data[10, 1] += 100; data[150, 0] += 25; data[170, 1] += -1e9;  # add artifacts
    >>> plotTimeSeries(data, ax=ax, ch_names=["Fz", "Cz"], color="red", zorder=0)
    >>> plt.legend(["clean", "_nolegend_", "with artefacts", "_nolegend_"])  # legend require to hide
    >>> plt.show()
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

    ch_names = is_valid_ch_names(ch_names, n_channels)

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
    times = np.linspace(offset, offset + (n_samples-1) / sfreq, num=n_samples)

    # compute shift based on scalings
    ax.plot(times, data, **kwargs)
    plt.yticks(-shifts, ch_names)
    plt.xlim(np.min(times), np.max(times))
    plt.ylim(np.min(-shifts)-(1.5*scalings), 1.5*scalings)
    return fig, ax


def assert_y_labels_correct(data, expected_labels):
    """Return assert error if the ax is not coming from data

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
        >>> annotations = [{'onset': 0.5, 'duration': 1.0, 'description': "blink", 'orig_time': 0.0},
        >>>                {'onset': 1., 'duration': 1.0, 'description': "blink", 'orig_time': 0.0}]
        or
        >>> # a list of two annotations starting after 0.5 and 1 second of duration 1.0 second named 'blink'
        >>> annotations = {'onset': [0.5, 1.0], 'duration': [1.0, 1.0],
        >>>                'description': ["blink", "blink"], 'orig_time': [0., 0.]}
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
                    if key not in ['onset', 'duration', 'description', 'orig_time']:
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
        if annotation["orig_time"] is None:
            annotation["orig_time"] = 0.0
        xmin = annotation["orig_time"] + annotation["onset"]
        xmax = annotation["orig_time"] + xmin + annotation["duration"]
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


def regression_report_with_plot(data, variables_x_axis, variables_y_axis, conditions=None, title=None):
    """Make regression on different variable of the DataFrame

    effect_variable = ["mask_delta", "mask_per", "VAS_I_delta", "VAS_I_per", "VAS_L_delta", "VAS_L_per"]
    Parameters
    ----------
    data: DataFrame
        all the data with columns being either values for regression or condition
        use query before to remove unwanted data
        Example:
        >>> # remove control subjects and keep only THR_classif of 3.0
        >>> data = reports.query("category != 'control' & THR_classif==3")

    effect_variable: list
        list of keys (columns) for `data` to use as x_axis for the regression (columns' values should be float)
        Example
        >>> effect_variable = ["mask_delta", "mask_per", "VAS_I_delta", "VAS_I_per", "VAS_L_delta", "VAS_L_per"]

    quantitative_variables: list
        list of keys (columns) for `data` to use as y_axis for the regression (columns' values should be float)
        Example
        >>> ['Number of episodes per hour', 'Number of tonic episodes per hour', 'Mean duration of phasic episode']

    conditions: str (default: None)
        key (column) for `data` to use as a different regression (values should be discrete)

    Returns
    -------
    meta_results: DataFrame
        the statistical significance, with 'correlation', 'pvalue'
        Example
        >>> meta_results.query("pvalue < 0.05")  # get all regression under pval<0.05


    """
    meta_results = pd.DataFrame()
    if conditions is None:
        conditions_values = ["None"]
    elif isinstance(conditions, str):
        conditions_values = data[conditions].unique()
    else:
        raise ValueError("conditions should be str (column key) or None")

    # loop over all quantitative variables (y_axis)
    for y_axis in variables_y_axis:
        # loop on all classification results (each figure)
        for threshold in conditions_values:
            if conditions is None:
                data_loc = data
            else:
                data_loc = data[data[conditions] == threshold]

            f, axes = plt.subplots(1, len(variables_x_axis), figsize=(len(variables_x_axis) * 7, 6))
            if len(variables_x_axis)==1:
                axes = [axes]
            # loop on all effect variables (each subplot)
            for x_axis, ax in zip(variables_x_axis, axes):
                regression_result = scipy.stats.linregress(data_loc[x_axis].values, data_loc[y_axis].values)
                sns.regplot(x=x_axis, y=y_axis, data=data_loc, fit_reg=True, ax=ax)
                ax.set_xlim(min(data_loc[x_axis].values) - 0.1, max(data_loc[x_axis].values) + 0.1)
                if conditions is None:
                    tmp = {"x_axis": x_axis, "y_axis": y_axis}
                else:
                    tmp = {"x_axis": x_axis, "y_axis": y_axis, conditions: [threshold]}

                if conditions is not None:
                    if isinstance(threshold, (int, float, np.integer, np.float)):
                        textstr = f"condition {threshold:.1f}"
                    else:
                        textstr = f"condition {str(threshold)}"
                else:
                    textstr = ""

                for a, re in zip(regression_result._fields, regression_result):
                    textstr = textstr + "\n" + f"{a} {re:.2f} "
                    tmp[a] = [re]
                ax.set_title(title)

                # place patch
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=props)

                # save results
                meta_results = pd.concat([meta_results, pd.DataFrame(tmp)])
    return meta_results
