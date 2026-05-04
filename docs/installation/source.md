(source)=

# Building From Source

Building from source means that you clone (or fork) the **PyAutoGalaxy** GitHub repository and run **PyAutoGalaxy** from
there. Unlike `conda` and `pip` this provides a build of the source code that you can edit and change, to
contribute the development **PyAutoGalaxy** or experiment with yourself!

A large amount of **PyAutoGalaxy** functionality is contained in its parent projects:

**PyAutoFit** <https://github.com/rhayes777/PyAutoFit>

**PyAutoArray** <https://github.com/Jammy2211/PyAutoArray>

If you wish to build from source all code you may need to build from source these 3 additional
projects. We include below instructions for building just **PyAutoGalaxy** from source or building all projects.

## Building Only PyAutoGalaxy

We upgrade pip to ensure certain libraries install:

```bash
pip install --upgrade pip
```

First, clone (or fork) the **PyAutoGalaxy** GitHub repository:

```bash
git clone https://github.com/Jammy2211/PyAutoGalaxy
```

Next, install the **PyAuto** parent projects via pip:

```bash
pip install autoconf
pip install autofit
pip install autoarray
```

Next, install the **PyAutoGalaxy** dependencies via pip:

```bash
pip install -r PyAutoGalaxy/requirements.txt
```

Next, install the optional dependency numba via pip (see [this link](https://pyautogalaxy.readthedocs.io/en/latest/installation/numba.html) for a description of numba):

```bash
pip install numba
```

For unit tests to pass you will also need the following optional requirements:

```bash
pip install pynufft
```

If you are using a `conda` environment, add the source repository as follows:

\[NOTE: Certain versions of conda use the command `conda develop` (without a dash) instead of those shown below.\]

```bash
conda-develop PyAutoGalaxy
```

Alternatively, if you are using a Python environment include the **PyAutoGalaxy** source repository in your PYTHONPATH
(noting that you must replace the text `/path/to` with the path to the **PyAutoGalaxy** directory on your computer):

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/PyAutoGalaxy
```

Finally, check the **PyAutoGalaxy** unit tests run and pass (you may need to install pytest via `pip install pytest`):

```bash
cd /path/to/PyAutoGalaxy
python3 -m pytest
```

## Building All Projects

We upgrade pip to ensure certain libraries install:

```bash
pip install --upgrade pip
```

First, clone (or fork) all 4 GitHub repositories:

```bash
git clone https://github.com/rhayes777/PyAutoFit
git clone https://github.com/Jammy2211/PyAutoArray
git clone https://github.com/Jammy2211/PyAutoGalaxy
```

Next, install **PyAutoConf** via pip:

```bash
pip install autoconf
```

Next, install the source build dependencies of each project via pip:

```bash
pip install -r PyAutoFit/requirements.txt
pip install -r PyAutoArray/requirements.txt
pip install -r PyAutoGalaxy/requirements.txt
```

Next, install the optional dependency numba via pip (see [this link](https://pyautogalaxy.readthedocs.io/en/latest/installation/numba.html) for a description of numba):

```bash
pip install numba
```

For unit tests to pass you will also need the following optional requirements:

```bash
pip install -r PyAutoArray/optional_requirements.txt
```

If you are using a `conda` environment, add each source repository as follows:

\[NOTE: Certain versions of conda use the command `conda develop` (without a dash) instead of those shown below.\]

```bash
conda-develop PyAutoFit
conda-develop PyAutoArray
conda-develop PyAutoGalaxy
```

Alternatively, if you are using a Python environment include each source repository in your PYTHONPATH
(noting that you must replace the text `/path/to` with the path to the each directory on your computer):

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/PyAutoFit
export PYTHONPATH=$PYTHONPATH:/path/to/PyAutoArray
export PYTHONPATH=$PYTHONPATH:/path/to/PyAutoGalaxy
```

Finally, check the unit tests run and pass for each project (you may need to install pytest via `pip install pytest`):

```bash
cd /path/to/PyAutoFit
python3 -m pytest
cd ../PyAutoArray
python3 -m pytest
cd ../PyAutoGalaxy
python3 -m pytest
```
