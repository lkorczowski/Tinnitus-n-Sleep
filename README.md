[![build_badge](https://github.com/lkorczowski/Tinnitus-n-Sleep/workflows/build/badge.svg)](
https://github.com/lkorczowski/Tinnitus-n-Sleep/actions)
[![codecov](https://codecov.io/gh/lkorczowski/Tinnitus-n-Sleep/branch/master/graph/badge.svg)](
https://codecov.io/gh/lkorcczowski/Tinnitus-n-Sleep)
# Tinnitus-n-Sleep
Detecting events in sleeping tinnitus patients

Work for SIOPI, by Robin Guillard & Louis Korczowski (2020-)

## What this toolbox do

This toolbox aims at detecting specific physiological events during sleep of patient suffering of tinnitus. For now, the main focus is 
1. EMG: automatic bruxism evaluation using detection of burst in electromyography activity
2. MEMA: automatic middle-ear muscles activation using detection of pressure variation of the ear canal
 
## Repo Organization
To so so, this toolbox is organized in several modules and folders:

**TINNSLEEP**
- **config**: configuration for cross-module usage storing user-specific files, folders and data. It is *REQUIRED* to add your data folders (hardcoded) if you want to use the preconfigured script and notebooks. 
- **data**: load, prepare and annotate data (mainly .edf using `mne`)
- **utils**: a lot of useful methods for preparing data, labels by doing simple operations
- **signal**: signal processing and automatic artifact thresholding
- **classification**: classification methods to detects events and artifacts. The main method is *AMDT* (Adaptive Mean Amplitud Thresholding).
- **pipeline**: "ready-to-go" configured *ADMT* pipeline for event classification
- **events**: methods to build and differentiate events
- **reports**: automatic reporting system using all above

**SCRIPT**
- ``compute_results.py`` is a preconfigured script which should be used in command line with option (by default it **WON'T** overwrite exiting results). It will load all suitable data and use `data_info.csv` to infers which operation to do for which file. Requirements: (a) add your local data folders in `tinnsleep.config.Config` (b) have a configured `data_info.csv` for the each related file.

**NOTEBOOKS**
- **Bruxism_detection**: preconfigured notebook to classify and visualize classification outputs for Bruxism detection of one subject.
- **Bruxism_Inter_subject_analyze**: preconfigured notebook for group-level analyze of bruxism (results are computed using ``compute_results.py``)
- **Middle_ear_detection**: preconfigured notebook to classify and visualize classification outputs for middle ear muscles activity (MEMA) detection
- **Middle_ear_Inter_subject_analyze**: preconfigured notebook for group-level analyze of MEMA (results are computed using ``compute_results.py``)
- and more...

## Results

### Automatic Detection
As shown in `notebooks/demo_mema_detection.ipynb` (e.g. Figure 1), a standardized method is used for MEMA and EMG events detection using the `tinnsleep.pipeline` methods. Here is some explained results.

[![demo1](
./images/demo_adaptive-emg+mema.png)](
./images/demo_adaptive-emg+mema.png)
**Figure 1:** Detection of both MEMA (orange) and EM (blue) events using adapative scheme. In this situation, EMG baseline has abrutly changed but the adaptive thresholding is able to both detect the first burst of EMG and then adapte to the new baseline. Meanwhile a burst of several MEMA is detected.

Classification is not enough to categorize events (see Figure 2), the `tinnsleep.events` methods allow to differentiate between different types of events. To do so, events are labeled into `bursts` and `episodes`:
- `bursts` are continuous events which are classified using the `tinnsleep.classification` module.
- `episodes` are a succession of bursts which are merged together using *scoring* methods. Each `episode` is labeled thanks to the properties of his bursts into *phasic*, *tonic* or *mixed* events.

[![demo2](
./images/demo_pure_mema.png)](
./images/demo_pure_mema.png)
**Figure 2:** Detection of pure MEMA (orange) `episode` consisting of two distant `bursts`. This two bursts are merged thank to the use of `tinnsleep.events.scoring` methods which allows to labels and categorize detected events. 
 
### Configuration

There are several key parameters for classification (see Figure 3). We advice to used pre-configured AMDT pipeline using `tinnsleep.pipeline`.

**CLASSIFICATION**
- Window length: length of the sliding window for computer instantaneous power (default: 250ms for Bruxism, 1s for MEMA)
- Baseline: memory buffer for the computation of the baseline. By default, we advice to use the non-casual "left-hand"+"righ-hand" baseline (see `tinnsleep.pipeline` for a pre-configured example) 
- Threshold: thresholding detects events for which power is X times greater than adaptive baseline

 [![bruxism_category](
./images/category_bruxism.png)](
./images/category_bruxism.png)
**Figure 3:** Influence of thresholding parameter in AMDT for the detection of the number of bruxism episodes per hour for each patient category (base on VAS-L).

### Bruxism: early results

[![bruxism_masking](
./images/trend_bruxism_masking.png)](
./images/trend_bruxism_masking.png)
**Figure 4:** Relationship between the absolute difference of tinnitus masking volume between before sleep onset and after awakening (x-axis) and number of detected bruxism episodes per hour (y-axis). The line represents the trend and area the confidence interval.

[![bruxism_VAS-L](
./images/trend_bruxism_VAS-L.png)](
./images/trend_bruxism_VAS-L.png)
**Figure 5:** Relationship between the absolute difference of tinnitus subjective loudness between before sleep onset and after awakening (x-axis) and number of detected bruxism episodes per hour (y-axis). The line represents the trend and area the confidence interval.

[![bruxism_VAS-I](
./images/trend_bruxism_VAS-I.png)](
./images/trend_bruxism_VAS-I.png)
**Figure 6:** Relationship between the absolute difference of tinnitus subjective intrusiveness between before sleep onset and after awakening (x-axis) and number of detected bruxism episodes per hour (y-axis). The line represents the trend and area the confidence interval.

### MEMA

[![mema_masking](
./images/trend_mema_masking.png)](
./images/trend_mema_masking.png)
**Figure 7:** Relationship between the absolute difference of tinnitus masking volume between before sleep onset and after awakening (x-axis) and number of detected MEMA episodes per hour (y-axis). The line represents the trend and area the confidence interval.

[![mema_VAS-L](
./images/trend_mema_VAS-L.png)](
./images/trend_mema_VAS-L.png)
**Figure 7:** Relationship between the absolute difference of tinnitus subjective loudness between before sleep onset and after awakening (x-axis) and number of detected MEMA episodes per hour (y-axis). The line represents the trend and area the confidence interval.

[![mema_VAS-I](
./images/trend_mema_VAS-I.png)](
./images/trend_mema_VAS-I.png)
**Figure 7:** Relationship between the absolute difference of tinnitus subjective intrusiveness volume between before sleep onset and after awakening (x-axis) and number of detected MEMA episodes per hour (y-axis). The line represents the trend and area the confidence interval.

## Installation
The following steps must be performed on a Anaconda prompt console, or 
alternatively, in a Windows command console that has executed the 
`C:\Anaconda3\Scripts\activate.bat` command that initializes the `PATH` so that
the `conda` command is found.

1. Checkout this repository and change to the cloned directory
   for the following steps.

    ```
    git clone git@github.com:lkorczowski/Tinnitus-n-Sleep.git
    cd Tinnitus-n-Sleep
    ```
    
2. Create a virtual environment with all dependencies.

    ```
    $ conda env create -f environment.yaml
    $ conda activate tinnsleep-env
    ```
    
3. Activate the environment and install this package (optionally with the `-e` 
    flag).
    ```
    $ conda activate tinnsleep-env
    (tinnsleep-env)$ pip install -e .
    ```

4. (optional) If you have a problem with a missing package, add it to the `environment.yaml`, then:
    ```
    (tinnsleep-env)$ conda env update --file environment.yaml
    ```

5. (optional) If you want to use the notebook, we advice Jupyter Lab (already in requirements) with additional steps:
    ```
    $ conda activate tinnsleep-env
    # install jupyter lab 
    (tinnsleep-env)$ conda install -c conda-forge jupyterlab 
    (tinnsleep-env)$ ipython kernel install --user --name=tinnsleep-env  
    (tinnsleep-env)$ jupyter lab  # run jupyter lab and select tinnsleep-env kernel
    # quit jupyter lab with CTRL+C then
    (tinnsleep-env)$ conda install -c conda-forge ipympl
    (tinnsleep-env)$ conda install -c conda-forge nodejs 
    (tinnsleep-env)$ jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-matplotlib
    ```
   
    To test if widget is working if fresh notebook:
    ```
    import pandas as pd
    import matplotlib.pyplot as plt
    %matplotlib widget
    
    df = pd.DataFrame({'a': [1,2,3]})
    
    plt.figure(2)
    plt.plot(df['a'])
    plt.show()
    ```

## How to have Git working

If you have trouble using git, a [tutorial](HOWTO_GIT_GITHUB_SSH_PR.md) is available to describe :
- how to set git with github and ssh
- how to contribute, create a new repository, participate
- how to branch, how to pull-request
- several tips while using git

I love also the great [git setup guideline](https://github.com/bbci/bbci_public/blob/master/HACKING.markdown) written
 for [BBCI Toolbox](https://github.com/bbci/bbci_public), just check it out! 
 
## How to contribute

A (incomplete) [guideline](CONTRIBUTING.md) is proposed to understand how to contribute and understand the best
practices with branchs and Pull-Requests.

