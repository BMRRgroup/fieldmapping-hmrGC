# hmrGC: Hierarchical multi-resolution graph-cuts for water–fat(–silicone) separation

Python library and scripts to perform field-mapping and water-fat(-silicone) separation based on the proposed hierarchical multi-resolution graph-cut framework from the publication:

*Jonathan K. Stelter, Christof Boehm, Stefan Ruschke, Kilian Weiss, Maximilian N. Diefenbach, Mingming Wu, Tabea Borde, Georg P. Schmidt, Marcus R. Makowski, Eva M. Fallenberg and Dimitrios C. Karampinos; Hierarchical multi-resolution graph-cuts for water–fat–silicone separation in breast MRI, IEEE Transactions on Medical Imaging, DOI: 10.1109/TMI.2022.3180302, https://ieeexplore.ieee.org/document/9788478*

## Python library for field-mapping

### Requirements

Development was performed using python-3.8. Requirements are stated in `setup.py`:
* numba (v0.55.0 recommended)
* pymaxflow (v1.2.13 recommended)
* opencv-python (v4.5.5.64 recommended)
* scipy (v1.7.3 recommended)
* *for GPU matrix computations:* cupy (v9.5.0 recommended)

Unit tests are stored in `/tests`. Pytest and h5py are needed as addtional requirements to run the tests.

### Installing

The package can be easily installed using Pip:

Direct installation from GitHub:
```
pip install git+https://github.com/BMRRgroup/fieldmapping-hmrGC
```

or clone the repository to use the developement mode (recommended):
```
git clone https://github.com/BMRRgroup/fieldmapping-hmrGC
pip install -e fieldmapping-hmrgc
```

### Quick start example
```
from hmrGC.dixon_imaging import MultiEcho

# Input arrays and parameters
signal = ...   # complex array with dim (nx, ny, nz, nte)
mask = ...   # boolean array with dim (nx, ny, nz)
params = {}
params['TE_s'] = ...   # float array with dim (nte)
params['centerFreq_Hz'] = ...   # float
params['fieldStrength_T'] = ...   # float
params['voxelSize_mm'] = ...   # recon voxel size with dim (3)
params['FatModel'] = {}
params['FatModel']['freqs_ppm'] = ...   # chemical shift difference between fat and water peak, float array with dim (nfatpeaks)
params['FatModel']['relAmps'] = ...   # relative amplitudes for each fat peak, float array with dim (nfatpeaks)
params['siliconePeak_ppm'] = ...   # only for water-fat-silicone separation, chemical shift difference between silicone and water peak, float

# Initialize MultiEcho object
g = MultiEcho(signal, mask, params)
g.r2star_correction = False   # modify runtime options, e.g. R2star correction for images

# Perform graph-cut method
g.perform()   # methods with different parameters can be defined using the multi_echo.json file
    
# Access separation results
fieldmap = g.fieldmap
r2starmap = g.r2starmap
waterimg = g.images['water']
fatimg = g.images['fat']
siliconeimg = g.images['silicone']   # only if silicone implants are present
pdffmap = g.images['fatFraction_percent']
```

## Publication reproducibility

Jupyter notebooks to reproduce the article can be found in `/publication`. Additional libraries are required and are specified in `/publication/requirements.txt`. You also need to clone all submodules: `git submodule update --init --recursive`. Phantom data are stored in `/publication/data` or can be downloaded from an alternative source (in case Github's LFS quote is exceed): https://syncandshare.lrz.de/getlink/fi2yT7Vp761X2EW2XbY41KnM/

## Authors and acknowledgment
**Main contributors:**
* Jonathan Stelter - [Body Magnetic Resonance Research Group, TUM](http://bmrr.de)
* Christof Boehm - [Body Magnetic Resonance Research Group, TUM](http://bmrr.de)

**Noise performance analysis using a generalized chemical species separation: https://github.com/BMRRgroup/MR_CSS**

**Single-voxel spectroscopy processing: https://github.com/BMRRgroup/alfonso**

## License
This project builds up on the [PyMaxflow](https://github.com/pmneila/PyMaxflow) library and the [Maxflow C++ implementation](https://pub.ist.ac.at/~vnk/software.html) by Yuri Boykov and Vladimir Kolmogorov and is therefore released under the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.
