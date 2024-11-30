# Galaxy_Power_Spectrum
Constrain the growth rate of the cosmic Large-Scale-Structure using auto- & cross- Power Spectrum of Galaxy density & momentum field. If you have any quesions or see any bug of the code, please contact me: qin@cppm.in2p3.fr

This is the final definitive edition. 

The code used to measure and model and fit the density and momentum auto- and cross- power spectrum multipoles of galaxy and velocity surveys. We have used 6dFGS and 6dfGSv mocks to test the code, we have 600 mocks for each survey. 

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

The following Python packages need to be installed if you want to use CosmPSPy to measure power spectrum:

1: numpy

2: scipy

\

The following Fortran packages need to be installed if you want to use CosmPSFt to measure power spectrum:

1: GSL : https://macappstore.org/gsl/ 

2: FFTW3 : https://www.fftw.org/install/mac.html

\

The following Python packages need to be installed if you want to calculate model power spectrum:

1: camb (used to calculate the mater power spectrum) : https://pypi.org/project/camb/

2: pypower (optional, only for window function convolution of Beuteler method) : https://github.com/cosmodesi/pypower

\

The following Python packages need to be installed if you want to fit growth rate:

1: ChainConsumer : https://samreay.github.io/ChainConsumer/

2: emcee 


