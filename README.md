# Galaxy_Power_Spectrum
Cosmological power spectrum of galaxies. Auto density power spectrum, auto momentum power spectrum and cross-power spectrum of the density and momentum fields.

Power spectrum measurements, modelling, window function convolution and fitting for the growth rate of the large-scale structure.  

The users will need to cite our papers:

Paper 1: Howlett 2019: https://ui.adsabs.harvard.edu/abs/2019MNRAS.487.5209H/abstract

Paper 2: Qin et al. 2019b: https://ui.adsabs.harvard.edu/abs/2019MNRAS.487.5235Q/abstract

Paper 3: Qin et al. 2025: https://ui.adsabs.harvard.edu/abs/2025ApJ...978....7Q/abstract

Paper 4: Qin et al. 2026: 

 

\
\
#################################################

CosmPSPy: 

The PYTHON code for power spectrum. The example for how to use the code can be found in 

https://github.com/FeiQin-cosmologist/Galaxy_Power_Spectrum/blob/main/CosmPSPy/Code/Examp_PS.ipynb

To install the PYTHON package 'CosmPSPy', the users can simply download the 'PSestFun.py' and 'PSmodFun.py' from 

https://github.com/FeiQin-cosmologist/Galaxy_Power_Spectrum/tree/main/CosmPSPy/Code 

to their laptop, then importing them using:

from PSestFun import *

from PSmodFun import *

So easy !!! The "PSestFun.py" is used to measure the power spectrum, while the "PSmodFun.py" is used to model the power spectrum and calculate the window function convolution. The 'CosmPSPy' can be downloaded from the GitHub link: 

https://github.com/FeiQin-cosmologist/Galaxy_Power_Spectrum/tree/main/CosmPSPy . 

The following Python packages are required to be installed for our code:

1: numpy

2: scipy 

You may also need CAMB (not compulsory) for calculating the linear matter power spectrum PL(k). You can use other methods to calculate the linear matter power spectrum.

3: CAMB : https://pypi.org/project/camb/

You may also need PYPOWER (not compulsory) for window function convolution (only for the 'Pypower' Method)

4: PYPOWER : https://pypower.readthedocs.io/en/latest/

\
\

#################################################

CosmPSFt: 

For the Fortran users, we have the corresponding Fortran code for power spectrum measurements (not for modelling). 
The following Fortran packages are required to be installed for our code:

1: fftw-3.3.10 : https://www.fftw.org/download.html

2: gsl-2.8 : https://formulae.brew.sh/formula/gsl 

\
\

#################################################

Math: 

The Mathematica code and manuscript draft used to derived the equations for power spectrum, including model power spectrum, power spectrum estimator and window function.

\
\

#################################################

FitExamp: 

We measure the power spectrum of 600 mocks of 6dFGS and 6dFGSv surveys. We fit the growth rate 'fsigma8' by comparing the model power spectrum to the measured power spectrum. We use density power spectrum, momentum power spectrum and cross-power spectrum. 

