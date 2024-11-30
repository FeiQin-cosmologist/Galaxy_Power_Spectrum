#!/bin/bash
 
 
gfortran -I/usr/local/include -L/usr/local/lib /Users/fei/WSP/Scie/CosmPS/CosmPSFt/PSrand.f90 /Users/fei/WSP/Scie/CosmPS/CosmPSFt/PSestFun.f90  -O3 -o Proj -lm -lgsl -lgslcblas -lfftw3
./Proj
gfortran -I/usr/local/include -L/usr/local/lib /Users/fei/WSP/Scie/CosmPS/CosmPSFt/PSdata.f90 /Users/fei/WSP/Scie/CosmPS/CosmPSFt/PSestFun.f90  -O3 -o Proj -lm -lgsl -lgslcblas -lfftw3
./Proj
