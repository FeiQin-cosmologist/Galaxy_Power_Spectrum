# Galaxy_Power_Spectrum
Constrain the growth rate of the cosmic Large-Scale-Structure using Galaxy Power Spectrum.

The code used to measure the density and momentum auto- and cross- power spectrum multipoles of galaxy and velocity surveys.  

Users will need to cite our papers:

Paper 1: Howlett 2019: https://ui.adsabs.harvard.edu/abs/2019MNRAS.487.5209H/abstract

Paper 2: Qin et al. 2019b: https://ui.adsabs.harvard.edu/abs/2019MNRAS.487.5235Q/abstract

Paper 3: Qin et al. 2024: https://arxiv.org/abs/2411.09571

Window Function Convolution: Blake et al. 2018: https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.5168B/abstract



CosmPSFt: measure power spectrum, Fortran code.

CosmPSPy: measure power spectrum, Python code. And the models of power spectrum and window function convolution are in this file. 

MathModelPS: the code to dirive the expressions of the models of power spectrum.



The following Python packages need to be installed if you want use to CosmPSPy to measure power spectrum:

1: numpy

2: scipy

3: camb (optional, used to calculate the mater power spectrum)

4: pypower (optional, only for window function convolution of Beuteler method)

5: ChainConsumer (optional, only for making plots)

6: emcee  (optional, only for the fit of growth rate)

The following Fortran packages need to be installed if you want use to CosmPSFt to measure power spectrum:

1: GSL

2: FFTW3


