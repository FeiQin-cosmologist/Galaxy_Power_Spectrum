# Galaxy_Power_Spectrum
This code is used to measure and model the density and momentum auto- and cross- power spectrum multipoles of galaxy and velocity surveys, as well as the window function convolution and constrain the growth rate. If you have any quesions or see any bug of the code, please contact me: qin@cppm.in2p3.fr

This is the final definitive edition. 

\

The examples for the utilization of this Code are shown in the following Jupyter notebook:

https://github.com/FeiQin-cosmologist/Galaxy_Power_Spectrum/blob/main/CosmPSPy/Code/Examp_PS.ipynb  

You can download the whole file 'CosmPSPy' to your laptop to run the notebook!!!

\

To install this code, the users can simply download the 'PSestFun.py' and 'PSmodFun.py' in the CosmPSPy file 

https://github.com/FeiQin-cosmologist/Galaxy_Power_Spectrum/tree/main/CosmPSPy/Code 

to their laptop, then importing them using:

from PSestFun import *

from PSmodFun import *

So easy !!! The following Python packages need to be installed if you want to use CosmPSPy to measure power spectrum:

1: numpy

2: scipy


\

Users will need to cite our papers:

Paper 1: Howlett 2019: https://ui.adsabs.harvard.edu/abs/2019MNRAS.487.5209H/abstract

Paper 2: Qin et al. 2019b: https://ui.adsabs.harvard.edu/abs/2019MNRAS.487.5235Q/abstract

Paper 3: Qin et al. 2024: https://arxiv.org/abs/2411.09571

Window Function Convolution: Blake et al. 2018: https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.5168B/abstract

\

CosmPSFt: measure power spectrum, Fortran code.

CosmPSPy: measure power spectrum, Python code. And the models of power spectrum and window function convolution are in this file. 

MathModelPS: the code to dirive the expressions of the models of power spectrum.

\



The following Fortran packages need to be installed if you want to use CosmPSFt to measure power spectrum:

1: GSL : https://macappstore.org/gsl/ 

2: FFTW3 : https://www.fftw.org/install/mac.html

\

The following Python packages may need to be installed if you want to calculate model power spectrum:

1: camb (optional, used to calculate the linear matter power spectrum) : https://pypi.org/project/camb/

2: pypower (optional, only for window function convolution of Beuteler method) : https://github.com/cosmodesi/pypower




