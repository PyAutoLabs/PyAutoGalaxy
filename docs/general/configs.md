# Configs

**PyAutoGalaxy** uses a number of configuration files that customize the default behaviour of the non-linear searches,
visualization and other aspects of **PyAutoGalaxy**.

Descriptions of every configuration file and their input parameters are provided in the `README.rst` in
the [config directory of the workspace](https://github.com/Jammy2211/autogalaxy_workspace/tree/release/config)

## Setup

By default, **PyAutoGalaxy** looks for the config files in a `config` folder in the current working directory, which is
why we run autogalaxy scripts from the `autogalaxy_workspace` directory.

The configuration path can also be set manually in a script using the project **PyAutoConf** and the following
command (the path to the `output` folder where the results of a non-linear search are stored is also set below):

```bash
from autoconf import conf

conf.instance.push(
    config_path="path/to/config",
    output_path=f"path/to/output"
)
```
