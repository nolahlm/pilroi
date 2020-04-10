# pilatus_roi

`pilatus_roi` is a python project that aids in the analysis and extraction of data from pilatus 2D detector images generated at SSRL beamlines 2-1 and 7-2 (and likely others...).  
This module is built on a main pandas dataframe `scan` with many functions acting on this DataFrame to reduce and visualize the dataset.

There are several modules:
* `data` handles data importing as well as data preprocessing (cropping, applying filters).  Key function is `create_scan()`
* `pilroi` contains functions used for creation and manipulation of scan() data via custom regions of interest
* `plotting` contains plotting stuff




Notebook todos:

* Better way to track median intenstiy max pixel
  * On top of this, should I instead input the "center" pixel as determined by Bart?