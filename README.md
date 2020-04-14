[![build_badge](https://github.com/lkorczowski/Tinnitus-n-Sleep/workflows/build/badge.svg)](
https://github.com/lkorczowski/Tinnitus-n-Sleep/actions)
[![codecov](https://codecov.io/gh/lkorczowski/Tinnitus-n-Sleep/branch/master/graph/badge.svg)](
https://codecov.io/gh/lkorcczowski/Tinnitus-n-Sleep)
# Tinnitus-n-Sleep
Detecting events in sleeping tinnitus patients
Work for SIOPI, by Robin Guillard & Louis Korczowski

## Installation
The following steps must be performed on a Anaconda prompt console, or 
alternatively, in a Windows command console that has executed the 
`C:\Anaconda3\Scripts\activate.bat` command that initializes the `PATH` so that
the `conda` command is found.

1. Checkout this repository and change to the cloned directory
   for the following steps.

    ```
    $ git clone git@github.com:lkorczowski/Tinnitus-n-Sleep.git
    $ cd Tinnitus-n-Sleep
    ```
    
2. Create a virtual environment with all dependencies.

    ```
    $ conda env create -f environment.yaml
    ```
    
3. Activate the environment and install this package (optionally with the `-e` 
    flag).

    ```
    $ conda activate tinnsleep-env
    $ pip install -e .
    ```

4. If you have a problem with a missing package, add it to the `environment.yaml`, then:
    ```
    $ conda env update --file environment.yaml
    ```
   
## How to have Git working

If you have trouble using git, a [tutorial](HOWTO_GIT_GITHUB_SSH_PR.md) is available to describe :
- how to set git with github and ssh
- how to contribute, create a new repository, participate
- how to branch, how to pull-request
- several tips while using git

## How to contribute

A (incomplete) [guideline](CONTRIBUTING.md) is proposed to understand how to contribute and understand the best
practices with branchs and Pull-Requests.

