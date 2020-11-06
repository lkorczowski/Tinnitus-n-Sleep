[![build_badge](https://github.com/lkorczowski/Tinnitus-n-Sleep/workflows/build/badge.svg)](
https://github.com/lkorczowski/Tinnitus-n-Sleep/actions)
[![codecov](https://codecov.io/gh/lkorczowski/Tinnitus-n-Sleep/branch/master/graph/badge.svg)](
https://codecov.io/gh/lkorcczowski/Tinnitus-n-Sleep)
# Tinnitus-n-Sleep
Detecting events in sleeping tinnitus patients

Work for SIOPI, by Robin Guillard & Louis Korczowski (2020-)

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

