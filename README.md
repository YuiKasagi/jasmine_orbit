# Jasmine_orbit

developer: Yui Kasagi

## Introduction

`Jasmine_orbit` is a tool for checking the visibility of a target and estimating thermal input conditions.

This project is largely based on RPR-SJ4B0509 (by H. Kataza).

For attitude during non-observation periods, RPR-SJ512017B was referenced, and the corresponding logic in OrbitAttitude.py was modified accordingly.

For more details, please refer to the document ``JASMINE-C2-TN-YKS-20251217-01-thermal_input'' on the JASMINE Wiki (accessible to JASMINE team members only).

## Installation

### Requirements
- Python 3.9 or later

### Install from GitHub

You can install the package directly from GitHub using `pip`:

```bash
pip install git+https://github.com/YuiKasagi/jasmine_orbit.git
```
Alternatively, clone the repository and install it locally:
```bash
git clone https://github.com/YuiKasagi/jasmine_orbit.git
cd jasmine_orbit
pip install .
```

## Usage

### Configuration

This package ships with **built-in default parameters**.  
All default values (thresholds, angle limits, etc.) are defined in:

- `src/jasmine_orbit/defaults.py`

You **do not need to edit** files under `src/` for normal use.

#### User configuration (recommended)

If you want to customize parameters (e.g., output directories, thresholds, catalog paths), create your own configuration file in your working directory (or anywhere you like), for example:

- `config/example_settings.py`

Then pass your config to functions/classes that support it:

```python
from config.example_settings import CONFIG
from jasmine_orbit.OrbitAttitude import calc_attitude

result = calc_attitude(..., config=CONFIG)
```
This approach keeps the library code immutable while allowing each user/project to manage settings in a reproducible way.

### Estimating Thermal Input to the Radiator Panel

#### Example

The following command performs a calculation for GJ 3929, starting 45 days after the vernal equinox, running for 90 days, and outputs both figures and data:

```
python main_target.py -s -p 45.0 -w 90. -o -t GJ 3929 
```

#### Command-line Options

```
usage:
    main_target.py [-h|--help] (-s|-a) -p <day_offset> -w <days> [-o] [-t <target_name>] [-m <minutes>]

options:
    -h --help           show this help message and exit
    -s                  use the vernal equinox as the reference date
    -a                  use the autumnal equinox as the reference date
    -p <day_offset>     calculation start day offset from the reference date (inclusive)
    -w <days>           calculation duration in days
    -o                  enable graph output (True or False)
    -t <target_name>    target name
    -m <minutes>        time step in minutes [default: 1]
```

## License
This project is released under the MIT License.
See the LICENSE file for full license text.