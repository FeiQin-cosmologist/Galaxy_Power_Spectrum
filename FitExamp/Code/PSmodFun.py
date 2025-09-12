import numpy as np
import scipy as sp
from scipy.special import sph_harm
from scipy.interpolate import splev 
from scipy import integrate
from scipy import interpolate, integrate
from scipy.special import gamma

 




###############################################################################   
######                                                                   ######
######                 Sec 1. Integration of linear Power                ######  
######                                                                   ######
############################################################################### 
# 1. Calculate integrations:     
def Fmn(m,n,r,x,y):
    if(m==0) and (n==0):
        FMN=(7.0*x + 3.0*r - 10.0*r*x*x)**2/(14.0*14.0*r*r*y*y*y*y);
    if(m==0) and (n==1):
        FMN= (7.0*x + 3.0*r - 10.0*r*x*x)*(7.0*x - r - 6.0*r*x*x)/(14.0*14.0*r*r*y*y*y*y);
    if(m==0) and (n==2):
        FMN= (x*x - 1.0)*(7.0*x + 3.0*r - 10.0*r*x*x)/(14.0*r*y*y*y*y);
    if(m==0) and (n==3):
        FMN= (1.0 - x*x)*(3.0*r*x - 1.0)/(r*r*y*y);       
    if(m==1) and (n==0):
        FMN= x*(7.0*x + 3.0*r - 10.0*r*x*x)/(14.0*r*r*y*y);
    if(m==1) and (n==1):
        FMN= (7.0*x - r - 6.0*r*x*x)**2/(14.0*14.0*r*r*y*y*y*y);
    if(m==1) and (n==2):
        FMN= (x*x - 1.0)*(7.0*x - r - 6.0*r*x*x)/(14.0*r*y*y*y*y);
    if(m==1) and (n==3):
        FMN=( 4.0*r*x + 3.0*x*x - 6.0*r*x*x*x - 1.0)/(2.0*r*r*y*y);             
    if(m==2) and (n==0):
        FMN= (2.0*x + r - 3.0*r*x*x)*(7.0*x + 3.0*r - 10.0*r*x*x)/(14.0*r*r*y*y*y*y);
    if(m==2) and (n==1):
        FMN= (2.0*x + r - 3.0*r*x*x)*(7.0*x - r - 6.0*r*x*x)/(14.0*r*r*y*y*y*y);
    if(m==2) and (n==2):
        FMN= x*(7.0*x - r - 6.0*r*x*x)/(14.0*r*r*y*y);
    if(m==2) and (n==3):
        FMN= 3.0*(1.0-x*x)*(1.0-x*x)/(y*y*y*y); 
    if(m==3) and (n==0):
        FMN= (1.0 - 3.0*x*x - 3.0*r*x + 5.0*r*x*x*x)/(r*r*y*y);
    if(m==3) and (n==1):
        FMN=  (1.0 - 2*r*x)*(1.0 - x*x)/(2.0*r*r*y*y);
    if(m==3) and (n==2):
        FMN=  (1.0 - x*x)*(2.0 - 12.0*r*x - 3.0*r*r + 15.0*r*r*x*x)/(r*r*y*y*y*y);
    if(m==3) and (n==3):
        FMN=  (-4.0 + 12.0*x*x + 24.0*r*x - 40.0*r*x*x*x + 3.0*r*r - 30.0*r*r*x*x + 35.0*r*r*x*x*x*x)/(r*r*y*y*y*y);        
    return FMN

def Imn_inner_integ(x,m,n,k,r,Pl_spline):
    y    = np.sqrt(1.0+r*r-2.0*r*x)#x=cos theta
    Plkq = sp.interpolate.splev(np.log10(k*y), Pl_spline, der=0)
    return Fmn(m,n,r,x,y)*10.0**(Plkq)

def Imn_outer_integ(r,m,n,k,Pl_spline):
    integ,errin= sp.integrate.quad(Imn_inner_integ,-1.0,1.0,epsabs=0.0,epsrel=1.0e-4,args=(m,n,k,r,Pl_spline))
    Pl   = sp.interpolate.splev(np.log10(k*r),Pl_spline, der=0)
    return r*r*integ*10.0**(Pl)

def I_mn(kmod,Pl_spline,ks):
    Imn=np.zeros((len(ks),4,4))
    for i in range(len(ks)):
        I00,errin= sp.integrate.quad(Imn_outer_integ,kmod[0],kmod[-1],args=(0,0,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        Imn[i,0,0]= I00*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)
        I01,errin= sp.integrate.quad(Imn_outer_integ,kmod[0],kmod[-1],args=(0,1,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        Imn[i,0,1]= I01*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)
        I02,errin= sp.integrate.quad(Imn_outer_integ,kmod[0],kmod[-1],args=(0,2,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        Imn[i,0,2]= I02*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)
        I03,errin= sp.integrate.quad(Imn_outer_integ,kmod[0],kmod[-1],args=(0,3,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        Imn[i,0,3]= I03*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)

        I10,errin= sp.integrate.quad(Imn_outer_integ,kmod[0],kmod[-1],args=(1,0,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        Imn[i,1,0]= I10*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)
        I11,errin= sp.integrate.quad(Imn_outer_integ,kmod[0],kmod[-1],args=(1,1,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        Imn[i,1,1]= I11*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)
        I12,errin= sp.integrate.quad(Imn_outer_integ,kmod[0],kmod[-1],args=(1,2,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        Imn[i,1,2]= I12*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)
        I13,errin= sp.integrate.quad(Imn_outer_integ,kmod[0],kmod[-1],args=(1,3,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        Imn[i,1,3]= I13*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)

        I20,errin= sp.integrate.quad(Imn_outer_integ,kmod[0],kmod[-1],args=(2,0,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        Imn[i,2,0]= I20*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)
        I21,errin= sp.integrate.quad(Imn_outer_integ,kmod[0],kmod[-1],args=(2,1,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        Imn[i,2,1]= I21*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)
        I22,errin= sp.integrate.quad(Imn_outer_integ,kmod[0],kmod[-1],args=(2,2,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3)
        Imn[i,2,2]= I22*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)
        I23,errin= sp.integrate.quad(Imn_outer_integ,kmod[0],kmod[-1],args=(2,3,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        Imn[i,2,3]= I23*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)

        I30,errin= sp.integrate.quad(Imn_outer_integ,kmod[0],kmod[-1],args=(3,0,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        Imn[i,3,0]= I30*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)
        I31,errin= sp.integrate.quad(Imn_outer_integ,kmod[0],kmod[-1],args=(3,1,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3)  
        Imn[i,3,1]= I31*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)
        I32,errin= sp.integrate.quad(Imn_outer_integ,kmod[0],kmod[-1],args=(3,2,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        Imn[i,3,2]= I32*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)
        I33,errin= sp.integrate.quad(Imn_outer_integ,kmod[0],kmod[-1],args=(3,3,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        Imn[i,3,3]= I33*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)
    return Imn

# 1.2:  -----------------------------------------------------------------------
def Gmn(m,n,r):    
    if(m==0) and (n==0):
        GMN= (12.0/(r*r) - 158.0 + 100.0*r*r - 42.0*r*r*r*r + (3.0/(r*r*r))*(r*r - 1.0)*(r*r - 1.0)*(r*r - 1.0)*(7.0*r*r + 2.0)*np.log((r + 1.0)/np.abs(r - 1.0)))/3024.0;
    if(m==0) and (n==1):
        GMN= (24.0/(r*r) - 202.0 + 56.0*r*r - 30.0*r*r*r*r + (3.0/(r*r*r))*(r*r - 1.0)*(r*r - 1.0)*(r*r - 1.0)*(5.0*r*r + 4.0)*np.log((r + 1.0)/np.abs(r - 1.0)))/3024.0;
    if(m==0) and (n==2):
        GMN= (2.0*(r*r + 1.0)*(3.0*r*r*r*r - 14.0*r*r + 3.0)/(r*r) - (3.0/(r*r*r))*(r*r - 1.0)*(r*r - 1.0)*(r*r - 1.0)*(r*r - 1.0)*np.log((r + 1.0)/np.abs(r - 1.0)))/224.0;
    if(m==1) and (n==0):
        GMN= (-38.0 +48.0*r*r - 18.0*r*r*r*r + (9.0/r)*(r*r - 1.0)*(r*r - 1.0)*(r*r - 1.0)*np.log((r + 1.0)/np.abs(r - 1.0)))/1008.0;
    if(m==1) and (n==1):
        GMN= (12.0/(r*r) - 82.0 + 4.0*r*r - 6.0*r*r*r*r + (3.0/(r*r*r))*(r*r - 1.0)*(r*r - 1.0)*(r*r - 1.0)*(r*r + 2.0)*np.log((r + 1.0)/np.abs(r - 1.0)))/1008.0;
    if(m==2) and (n==0):
        GMN= (2.0*(9.0 - 109.0*r*r + 63.0*r*r*r*r - 27.0*r*r*r*r*r*r)/(r*r) + (9.0/(r*r*r))*(r*r - 1.0)*(r*r - 1.0)*(r*r - 1.0)*(3.0*r*r + 1.0)*np.log((r + 1.0)/np.abs(r - 1.0)))/672.0;
    return GMN

def Jmn_integ(q,m,n,k,Pl_spline):
    r    = q/k
    if(r==1.0):
        r=(q+0.00001*q)/k
    Pl   = sp.interpolate.splev(np.log10(q),Pl_spline, der=0)
    return Gmn(m,n,r)*10.0**(Pl)

def J_mn(kmod,Pl_spline,ks):
    Jmn=np.zeros((len(ks),2,3))
    for i in range(len(ks)):
        J00,errin= sp.integrate.quad(Jmn_integ,kmod[0],kmod[-1],args=(0,0,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3)
        Jmn[i,0,0] = J00/(2.0*np.pi*np.pi)
        J01,errin= sp.integrate.quad(Jmn_integ,kmod[0],kmod[-1],args=(0,1,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3)
        Jmn[i,0,1] = J01/(2.0*np.pi*np.pi)
        J02,errin= sp.integrate.quad(Jmn_integ,kmod[0],kmod[-1],args=(0,2,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        Jmn[i,0,2] = J02/(2.0*np.pi*np.pi)

        J10,errin= sp.integrate.quad(Jmn_integ,kmod[0],kmod[-1],args=(1,0,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        Jmn[i,1,0] = J10/(2.0*np.pi*np.pi)
        J11,errin= sp.integrate.quad(Jmn_integ,kmod[0],kmod[-1],args=(1,1,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        Jmn[i,1,1] = J11/(2.0*np.pi*np.pi)
        J20,errin= sp.integrate.quad(Jmn_integ,kmod[0],kmod[-1],args=(2,0,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3)
        Jmn[i,1,2] = J20/(2.0*np.pi*np.pi)
    return Jmn

# 1.3 -------------------------------------------------------------------------
def hmn(m,n,s,r,x,y):
    if (m == 0) and (n == 0):
        if (s == 0):
            numer = 7.0*x + 3.0*r - 10.0*r*x*x;
            return numer/(14.0*r*y*y);
        if (s == 1):
            numer = (7.0*x + 3.0*r - 10.0*r*x*x)*(3.0*x*x - 4.0*r*x + 2.0*r*r - 1.0);
            return numer/(14.0*3.0*r*y*y*y*y);
    if (m == 0) and (n == 1):
        if (s == 0):
            return 1.0;
        if (s == 1):
            numer = (3.0*x*x - 4.0*r*x + 2.0*r*r - 1.0);
            return (numer*numer)/(3.0*3.0*y*y*y*y);
    if (m == 0) and (n == 2):
        if (s == 1):
            numer = (3.0*x*x - 4.0*r*x + 2.0*r*r - 1.0);
            return numer/(3.0*y*y);
    if (m == 1) and (n == 0):
        if (s == 0):
            numer = 7.0*x - r - 6.0*r*x*x;
            return numer/(14.0*r*y*y);
        if (s == 1):
            numer = (7.0*x - r - 6.0*r*x*x)*(3.0*x*x - 4.0*r*x + 2.0*r*r - 1.0);
            return numer/(14.0*3.0*r*y*y*y*y);
    if (m == 1) and (n == 1):
        if (s == 0):
            return x/r;
        if (s == 1):
            numer = (3.0*x*x - 4.0*r*x + 2.0*r*r - 1.0)*x/r;
            return numer/(3.0*y*y);
    if (m == 2) and (n == 0):
       if (s == 0):
           numer = (x*x - 1.0)/2.0
           return numer/(y*y);
       if (s == 1):
           numer = (x*x - 1.0)*(3.0*x*x - 4.0*r*x + 2.0*r*r - 1.0);
           return numer/(6.0*y*y*y*y); 
    if (m == 3) and (n == 0):
       if (s == 0):
           numer = (2.0*x + r - 3.0*r*x*x)/2.0;
           return numer/(r*y*y);
       if (s == 1):
           numer = (2.0*x + r - 3.0*r*x*x)*(3.0*x*x - 4.0*r*x + 2.0*r*r - 1.0);
           return numer/(6.0*r*y*y*y*y); 

def Kmn_inner_integ(x,m,n,s,k,r,Pl_spline):
    y    = np.sqrt(1.0+r*r-2.0*r*x)
    Plkq = sp.interpolate.splev(np.log10(k*y), Pl_spline, der=0)
    return hmn(m,n,s,r,x,y)*10.0**(Plkq)

def Kmn_outer_integ(r,m,n,s,k,Pl_spline):
    integ,errin= sp.integrate.quad(Kmn_inner_integ,-1.0,1.0,epsabs=0.0,epsrel=1.0e-3,args=(m,n,s,k,r,Pl_spline))
    Pl   = sp.interpolate.splev(np.log10(k*r),Pl_spline, der=0)
    return r*r*integ*10.0**(Pl)

def K_mn(kmod,Pl_spline,ks):
    KMN=np.zeros((len(ks),3,2))
    KSmn=np.zeros((len(ks),7))
    for i in range(len(ks)):
        K00,errin = sp.integrate.quad(Kmn_outer_integ,kmod[0],kmod[-1],args=(0,0,0,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        KMN[i,0,0]= K00*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi) 
        Ks00,errin= sp.integrate.quad(Kmn_outer_integ,kmod[0],kmod[-1],args=(0,0,1,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        KSmn[i,0] = Ks00*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi) 
        
        K01,errin = sp.integrate.quad(Kmn_outer_integ,kmod[0],kmod[-1],args=(0,1,0,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        KMN[i,0,1]= K01*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)#-norm
        Ks01,errin= sp.integrate.quad(Kmn_outer_integ,kmod[0],kmod[-1],args=(0,1,1,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        KSmn[i,1] = Ks01*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)#-4.0/9.0*norm
        Ks02,errin= sp.integrate.quad(Kmn_outer_integ,kmod[0],kmod[-1],args=(0,2,1,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        KSmn[i,2] = Ks02*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)#-2.0/3.0*norm

        K10,errin = sp.integrate.quad(Kmn_outer_integ,kmod[0],kmod[-1],args=(1,0,0,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        KMN[i,1,0]= K10*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)
        Ks10,errin= sp.integrate.quad(Kmn_outer_integ,kmod[0],kmod[-1],args=(1,0,1,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        KSmn[i,3] = Ks10*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)
        
        K11,errin = sp.integrate.quad(Kmn_outer_integ,kmod[0],kmod[-1],args=(1,1,0,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        KMN[i,1,1]= K11*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi) 
        Ks11,errin= sp.integrate.quad(Kmn_outer_integ,kmod[0],kmod[-1],args=(1,1,1,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        KSmn[i,4] = Ks11*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)

        K20,errin = sp.integrate.quad(Kmn_outer_integ,kmod[0],kmod[-1],args=(2,0,0,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        KMN[i,2,0]= K20*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)
        Ks20,errin= sp.integrate.quad(Kmn_outer_integ,kmod[0],kmod[-1],args=(2,0,1,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        KSmn[i,5] = Ks20*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)
        
        K30,errin = sp.integrate.quad(Kmn_outer_integ,kmod[0],kmod[-1],args=(3,0,0,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        KMN[i,2,1]= K30*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)
        Ks30,errin= sp.integrate.quad(Kmn_outer_integ,kmod[0],kmod[-1],args=(3,0,1,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        KSmn[i,6] = Ks30*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)
    return KMN ,KSmn

# 1.4 -------------------------------------------------------------------------
def pk_integ(r,kval,Pl_spline):
    P_lin = sp.interpolate.splev(np.log10(r*kval),Pl_spline, der=0)
    return r*r*10.0**(P_lin)*10.0**(P_lin)

def KNORm(kmod,Pl_spline,ks): 
    xsa=np.zeros(len(ks))
    for i in range(len(ks)):
        integ,errin = sp.integrate.quad(pk_integ,kmod[0],kmod[len(kmod)-1],args=(ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3)
        xsa[i]=ks[i]*ks[i]*ks[i]*integ/(2.0*np.pi*np.pi)
    return xsa

def SIGMA3_inner_integ(x,r):
    y     = np.sqrt(1.0+r*r-2.0*r*x)
    numer = (2.0/7.0)*(x*x - 1.0)*(3.0*x*x - 4.0*r*x + 2.0*r*r - 1.0)
    return numer/(3.0*y*y) + 8.0/63.0
    
def SIGMA3_outter_integ(r,k,Pl_spline):  
    integ,errin= sp.integrate.quad(SIGMA3_inner_integ,-1.0,1.0,epsabs=0.0,epsrel=1.0e-4,args=(r))
    Pl   = sp.interpolate.splev(np.log10(k*r),Pl_spline, der=0)
    return integ*r*r*10.0**(Pl)

def SIGMA3(kmod,Pl_spline,ks):
    SIGMAs=np.zeros(len(ks))
    for i in range(len(ks)):
        integ,errin= sp.integrate.quad(SIGMA3_outter_integ,kmod[0],kmod[-1],args=(ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        SIGMAs[i]=integ*(105.0*ks[i]*ks[i]*ks[i])/(64.0*np.pi*np.pi)
    return SIGMAs

# 2. the parameter estimation main code:---------------------------------------
def CAMB_Fun(kmin,kmax,nk,RsFeff,hub,ombh2,omch2,ns, As, halofit=False):
    import camb
    from camb import model 
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=hub,ombh2=ombh2, omch2=omch2)
    pars.set_dark_energy()
    pars.InitPower.set_params(As=As,ns=ns)
    pars.set_matter_power(redshifts=[RsFeff], kmax=kmax)
    if(halofit==False):
        pars.NonLinear = model.NonLinear_none  
    else:
        pars.NonLinear = camb.model.NonLinear_both #(or NonLinear_lens, NonLinear_both)
        pars.NonLinearModel.set_params(halofit_version=halofit) # halofit = "mead2020" # https://camb.readthedocs.io/en/latest/nonlinear.html
    results = camb.get_results(pars)    
    kh, z, pk = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints = nk)
    return kh, pk[0]  , results.get_sigma8()
def PSloop_Fun( KmodLim,Nkmod,hub,ombh2,omch2,ns, As,out_dir ): 
    kh,pk,sig8=CAMB_Fun(1e-4,100.0,Nkmod,0,hub,ombh2,omch2,ns, As, False)
    ks=kh[kh<=KmodLim]
    print( 'sigma8_fid_integ = ',sig8,'\n')
    Pl_spline = sp.interpolate.splrep(np.log10(kh),np.log10(pk ), s=0)
    # integrations:
    print( '\n Integration-Imn \n')
    Imn=I_mn(kh,Pl_spline,ks)
    print( '\n Integration-Jmn \n')
    Jmn=J_mn(kh,Pl_spline,ks)
    print( '\n Integration-Kmn \n')
    Kmn,Ksmn=K_mn(kh,Pl_spline,ks)
    print( '\n Integration-Sigma3 \n')
    SIGMA_3=SIGMA3(kh,Pl_spline,ks)
    print( '\n Integration-norm \n')
    Intg_norm=KNORm(kh,Pl_spline,ks)
    # outputs:
    I00=Imn[:,0,0]   ; I01=Imn[:,0,1]   ; I02=Imn[:,0,2]   ; I03=Imn[:,0,3]
    I10=Imn[:,1,0]   ; I11=Imn[:,1,1]   ; I12=Imn[:,1,2]   ; I13=Imn[:,1,3]
    I20=Imn[:,2,0]   ; I21=Imn[:,2,1]   ; I22=Imn[:,2,2]   ; I23=Imn[:,2,3]
    I30=Imn[:,3,0]   ; I31=Imn[:,3,1]   ; I32=Imn[:,3,2]   ; I33=Imn[:,3,3]
    J00=Jmn[:,0,0]   ; J01=Jmn[:,0,1]   ; J02=Jmn[:,0,2]   ; J10=Jmn[:,1,0] 
    J11=Jmn[:,1,1]   ; J20=Jmn[:,1,2]       
    K00=Kmn[:,0,0]   ; K01=Kmn[:,0,1]   
    K10=Kmn[:,1,0]   ; K11=Kmn[:,1,1]
    K20=Kmn[:,2,0]   ; K30=Kmn[:,2,1]       
    Ks00=Ksmn[:,0]   ; Ks01=Ksmn[:,1]   ; Ks02=Ksmn[:,2]   ; Ks10=Ksmn[:,3]        
    Ks11=Ksmn[:,4]   ; Ks20=Ksmn[:,5]   ; Ks30=Ksmn[:,6]
    outfile    = open(out_dir+'INTEG_PL.npy', 'w')
    for i in range(len(ks)):
       outfile.write("%12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf\n"% (kh[i],pk[i],   I00[i],  I01[i],  I02[i],  I03[i],  I10[i],  I11[i],  I12[i],  I13[i],  I20[i],  I21[i],  I22[i],  I23[i],  I30[i],  I31[i],  I32[i],  I33[i],  J00[i],  J01[i],  J02[i],  J10[i],  J11[i],  J20[i],  K00[i],  Ks00[i], K01[i],  Ks01[i], Ks02[i], K10[i],  Ks10[i], K11[i],  Ks11[i], K20[i],  Ks20[i], K30[i],  Ks30[i], SIGMA_3[i],Intg_norm[i]))
    outfile.close()
    PT=np.loadtxt(out_dir+'INTEG_PL.npy')
    PT=PT.T
    PT[26,0:] -= PT[38,0:];PT[27,0:] -= 4.0/9.0*PT[38,0:];PT[28,0:] -= 2.0/3.0*PT[38,0:]
    np.save(out_dir+'INTEG_PL',np.array([PT,sig8],dtype=object),allow_pickle=True)    
    return PT,sig8,out_dir+'INTEG_PL.npy'
def PSloopInput_Fun( KmodLim,kh,pk,fid_sigma8,out_dir ): 
    ks=kh[kh<=KmodLim]
    Pl_spline = sp.interpolate.splrep(np.log10(kh),np.log10(pk ), s=0)
    # integrations:
    print( '\n Integration-Imn \n')
    Imn=I_mn(kh,Pl_spline,ks)
    print( '\n Integration-Jmn \n')
    Jmn=J_mn(kh,Pl_spline,ks)
    print( '\n Integration-Kmn \n')
    Kmn,Ksmn=K_mn(kh,Pl_spline,ks)
    print( '\n Integration-Sigma3 \n')
    SIGMA_3=SIGMA3(kh,Pl_spline,ks)
    print( '\n Integration-norm \n')
    Intg_norm=KNORm(kh,Pl_spline,ks)
    # outputs:
    I00=Imn[:,0,0]   ; I01=Imn[:,0,1]   ; I02=Imn[:,0,2]   ; I03=Imn[:,0,3]
    I10=Imn[:,1,0]   ; I11=Imn[:,1,1]   ; I12=Imn[:,1,2]   ; I13=Imn[:,1,3]
    I20=Imn[:,2,0]   ; I21=Imn[:,2,1]   ; I22=Imn[:,2,2]   ; I23=Imn[:,2,3]
    I30=Imn[:,3,0]   ; I31=Imn[:,3,1]   ; I32=Imn[:,3,2]   ; I33=Imn[:,3,3]
    J00=Jmn[:,0,0]   ; J01=Jmn[:,0,1]   ; J02=Jmn[:,0,2]   ; J10=Jmn[:,1,0] 
    J11=Jmn[:,1,1]   ; J20=Jmn[:,1,2]       
    K00=Kmn[:,0,0]   ; K01=Kmn[:,0,1]   
    K10=Kmn[:,1,0]   ; K11=Kmn[:,1,1]
    K20=Kmn[:,2,0]   ; K30=Kmn[:,2,1]       
    Ks00=Ksmn[:,0]   ; Ks01=Ksmn[:,1]   ; Ks02=Ksmn[:,2]   ; Ks10=Ksmn[:,3]        
    Ks11=Ksmn[:,4]   ; Ks20=Ksmn[:,5]   ; Ks30=Ksmn[:,6]
    outfile    = open(out_dir+'INTEG_PL.npy', 'w')
    for i in range(len(ks)):
       outfile.write("%12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf\n"% (kh[i],pk[i],   I00[i],  I01[i],  I02[i],  I03[i],  I10[i],  I11[i],  I12[i],  I13[i],  I20[i],  I21[i],  I22[i],  I23[i],  I30[i],  I31[i],  I32[i],  I33[i],  J00[i],  J01[i],  J02[i],  J10[i],  J11[i],  J20[i],  K00[i],  Ks00[i], K01[i],  Ks01[i], Ks02[i], K10[i],  Ks10[i], K11[i],  Ks11[i], K20[i],  Ks20[i], K30[i],  Ks30[i], SIGMA_3[i],Intg_norm[i]))
    outfile.close()
    PT=np.loadtxt(out_dir+'INTEG_PL.npy')
    PT=PT.T
    PT[26,0:] -= PT[38,0:];PT[27,0:] -= 4.0/9.0*PT[38,0:];PT[28,0:] -= 2.0/3.0*PT[38,0:]
    np.save(out_dir+'INTEG_PL',np.array([PT,fid_sigma8],dtype=object),allow_pickle=True)    
    return PT,fid_sigma8,out_dir+'INTEG_PL.npy'
#########################     The end of Sec 1.    ############################ 
############################################################################### 






















###############################################################################
#####                                                                     #####
#####                   Sec 2. The model power spectrum                   #####
#####                                                                     #####
###############################################################################
# 1. calculate Loop terms Pmn: 
def PijFun(parm,PT,MU,pstyp):
    if(pstyp=='auto'):
        f, b1,b2,bs,b3nl, sigv1_squre,sigv2_squre,sigv3_squre, Dz = parm
    if(pstyp=='crs'):  
        f, b1,b2,bs,b3nl, b1v,b2v,bsv,  sigvv1_squre,sigvv2_squre,sigvv3_squre,   Dz = parm  
    k  =PT[0,0:] ; Pl =PT[1,0:]   
    I00=PT[2,0:]  ;I01=PT[3,0:]  ;I02=PT[4,0:]  ;I03=PT[5,0:]  ;I10=PT[6,0:]  ;I11=PT[7,0:]  ;I12=PT[8,0:]  ;I13=PT[9,0:] ;
    I20=PT[10,0:] ;I21=PT[11,0:] ;I22=PT[12,0:] ;I23=PT[13,0:] ;I30=PT[14,0:] ;I31=PT[15,0:] ;I32=PT[16,0:] ;I33=PT[17,0:] ;
    J00=PT[18,0:] ;J01=PT[19,0:];J02=PT[20,0:]; J10=PT[21,0:]; J11=PT[22,0:];  J20=PT[23,0:]; K00=PT[24,0:]; Ks00=PT[25,0:];
    K01=PT[26,0:]; Ks01=PT[27,0:];Ks02=PT[28,0:];K10=PT[29,0:];Ks10=PT[30,0:]; K11=PT[31,0:];
    Ks11=PT[32,0:];K20=PT[33,0:];Ks20=PT[34,0:];K30=PT[35,0:];Ks30=PT[36,0:];  sig3_squre=PT[37,0:];#Intg_norm=PT[38,0:];   
    sig4_squre=1.0/(24.0*np.pi*np.pi)*(Dz**4*sp.integrate.simps( (I23+2./3.*I32+I33/5.) /k**2,k))    
    P_00=np.zeros((len(k),len(MU)));P_01=np.zeros((len(k),len(MU)));P_02=np.zeros((len(k),len(MU)))
    P_03=np.zeros((len(k),len(MU)));P_04=np.zeros((len(k),len(MU)));P_11=np.zeros((len(k),len(MU)))
    P_12=np.zeros((len(k),len(MU)));P_13=np.zeros((len(k),len(MU)));P_22=np.zeros((len(k),len(MU)))
    for ss in range(len(MU)):
        mu=MU[ss]
        if(pstyp=='auto'): 
            P00=(b1+b1)*Dz**4*(b2*K00+bs*Ks00)+b1*b1*Dz**2*(Pl+2.*Dz**2*(I00+3.*k*k*Pl*J00))+1./2.*Dz**4*(b2*b2*K01+bs*bs*Ks01+2.*b2*bs*Ks02+4.*b3nl*sig3_squre*Pl)
            P01=f*b1*Dz**2*(Pl+2.*Dz**2*(I01+b1*I10+3.*k*k*Pl*(J01+b1*J10))-b2*Dz**2*K11-bs*Dz**2*Ks11)-f*Dz**4*(b2*K10+bs*Ks10+b3nl*sig3_squre*Pl)
            P02=f*f*b1*Dz**4*(I02+mu**2*I20+2.*k*k*Pl*(J02+mu**2*J20))-f*f*k*k*sigv1_squre*P00+f*f*Dz**4*(b2*(K20+mu**2*K30)+bs*(Ks20+mu**2*Ks30))
            P03=-f*f*k*k*sigv2_squre*P01
            P04=-1./2.*f*f*f*f*b1*k*k*sigv3_squre*Dz**4*(I02+mu**2*I20+2.*k*k*Pl*(J02+mu**2*J20)) +1./4.*f*f*f*f*b1*b1*k**4*P00*(sigv3_squre**2+sig4_squre) 
            P11=f*f*Dz**2*(mu**2*(Pl+Dz**2*(2.*I11+(b1+b1+b1+b1)*I22+b1*b1*I13+6.*k**2*Pl*(J11+(b1+b1)*J10)))  +b1*b1*Dz**2*I31)
            P12=f*f*f*Dz**4*(I12+mu**2*I21-b1*(I03+mu**2*I30)+2.*k**2*Pl*(J02+mu**2*J20))   -f*f*k*k*sigv1_squre* P01+2.*f*f*f*k*k*Dz**4*(I01+I10+3.*k*k*Pl*(J01+J10))   *sigv1_squre 
            P13=-f*f*k*k*f*f*Dz**2*(sigv2_squre*mu**2*(Pl+Dz**2*(2.*I11+(b1+b1+b1+b1)*I22 +6.*k*k*Pl*(J11+(b1+b1)*J10))) +sigv1_squre*b1*b1*Dz**2*(mu**2*I13+I31) )
            P22=1./4.*f*f*f*f*Dz**4*(I23+2.*mu**2*I32+mu**4*I33)+f*f*f*f*k**4*sigv1_squre**2 *P00-f*f*k*k*sigv1_squre*(2.*P02-f*f*Dz**4*(b2*(K20+mu**2*K30)+bs*(Ks20+mu**2*Ks30)))          
        if(pstyp=='crs'):
            P00=b1*b1v*Pl*Dz**2+b1*Dz**4*(b2v*K00+bsv*Ks00+2.*b1v*(I00+3.*J00*k*k*Pl)) +1./2.*Dz**4*(b2*b2v*K01+2.*b1v*(b2*K00+bs*Ks00)+bs*bsv*Ks01+b2v*bs*Ks02+b2*bsv*Ks02+4.*b3nl*Pl*sig3_squre)
            P01=f*Dz**2*(b1*(Pl+2.*(I01+b1v*I10+3.*(J01+b1v*J10)*k*k*Pl)*Dz**2) -Dz**2*(b2*(K10+b1v*K11)+bs*Ks10+b1v*bs*Ks11+b3nl*Pl*sig3_squre))
            P11=f*f*Dz**2*(Pl*mu*mu+Dz**2*(b1*b1v*I31+(2.*I11+b1*b1v*I13+2.*b1*I22+2.*b1v*I22+6.*((b1+b1v)*J10+J11)*k*k*Pl)*mu*mu))
            P02=f*f*b1*Dz**4*(I02+mu**2*I20+2.*k*k*Pl*(J02+mu**2*J20))-f*f*k*k*sigvv1_squre*P00+f*f*Dz**4*(b2*(K20+mu**2*K30)+bs*(Ks20+mu**2*Ks30))
            P03=-f*f*k*k*sigvv2_squre*P01
            P04=-1./2.*f*f*f*f*b1*k*k*sigvv3_squre*Dz**4*(I02+mu**2*I20+2.*k*k*Pl*(J02+mu**2*J20)) +1./4.*f*f*f*f*b1*b1v*k**4*P00*(sigvv3_squre**2+sig4_squre) 
            P12=f*f*f*Dz**4*(I12+mu**2*I21-b1*(I03+mu**2*I30)+2.*k**2*Pl*(J02+mu**2*J20))   -f*f*k*k*sigvv1_squre* P01+2.*f*f*f*k*k*Dz**4*(I01+I10+3.*k*k*Pl*(J01+J10))   *sigvv1_squre 
            P13=-f*f*k*k*f*f*Dz**2*(sigvv2_squre*mu**2*(Pl+Dz**2*(2.*I11+(b1+b1+b1v+b1v)*I22 +6.*k*k*Pl*(J11+(b1+b1v)*J10))) +sigvv1_squre*b1*b1v*Dz**2*(mu**2*I13+I31) )
            P22=1./4.*f*f*f*f*Dz**4*(I23+2.*mu**2*I32+mu**4*I33)+f*f*f*f*k**4*sigvv1_squre**2 *P00-f*f*k*k*sigvv1_squre*(2.*P02-f*f*Dz**4*(b2*(K20+mu**2*K30)+bs*(Ks20+mu**2*Ks30)))          
        P_00[:,ss]=P00;   P_01[:,ss]=P01;   P_02[:,ss]=P02
        P_03[:,ss]=P03;   P_04[:,ss]=P04;   P_11[:,ss]=P11
        P_12[:,ss]=P12;   P_13[:,ss]=P13;  P_22[:,ss]=P22 
    return   P_00,P_01,P_02,P_03,P_04,P_11,P_12,P_13,P_22  

# 2. calculate model power spectrum:-------------------------------------------
def Pkmod_Fun(params,sigma8_fid,PT,ps_type, az,Hz,Dz, do_Multi=True, GaliTrans=False  ):  
    if(GaliTrans):
        fsigma8,  b1sigma8,b2sigma8,bssigma8,b3nlsigma8,  b1vsigma8,b2vsigma8,bsvsigma8,b3nlvsigma8,    sigmav1_squre,sigmav2_squre,sigmav3_squre,    sigmavv1_squre,sigmavv2_squre,sigmavv3_squre, epsilon  = params      
    else: 
        fsigma8,  b1sigma8,b2sigma8,bssigma8,b3nlsigma8,  b1vsigma8,b2vsigma8,bsvsigma8,b3nlvsigma8,    sigmav1_squre,sigmav2_squre,sigmav3_squre,    sigmavv1_squre,sigmavv2_squre,sigmavv3_squre  = params   
    sigmav1_squre ,sigmav2_squre ,sigmav3_squre  = np.abs(sigmav1_squre) ,np.abs(sigmav2_squre) ,np.abs(sigmav3_squre)   
    sigmavv1_squre,sigmavv2_squre,sigmavv3_squre = np.abs(sigmavv1_squre),np.abs(sigmavv2_squre),np.abs(sigmavv3_squre )  
    # normalize parms: 
    f = fsigma8/sigma8_fid     
    # density field:  
    b1        = b1sigma8/sigma8_fid 
    b2        = b2sigma8/sigma8_fid
    if(bssigma8=='anal'):
        bs    = -4.0/7.0*(b1-1.0)
    else:    
        bs    = bssigma8/sigma8_fid     
    if(b3nlsigma8=='anal'):
        b3nl  = 32.0/315.0*(b1-1.0)
    else:    
        b3nl  = b3nlsigma8/sigma8_fid 
    # momentum field: 
    b1v       = b1vsigma8/sigma8_fid
    b2v       = b2vsigma8/sigma8_fid
    if(bsvsigma8=='anal'):
        bsv   = -4.0/7.0*(b1v-1.0)
    else:    
        bsv   = bsvsigma8/sigma8_fid 
    if(b3nlvsigma8=='anal'):
        b3nlv  = 32.0/315.0*(b1v-1.0)
    else:    
        b3nlv  = b3nlvsigma8/sigma8_fid      
    # model PS:
    P_model=[];PSTYP=[];PSRSD=[] 
    mu = np.linspace(0.0, 1.0, 300) 
    if(not GaliTrans):
        if ('den' in ps_type):
            P_00,P_01,P_02,P_03,P_04,P_11,P_12,P_13,P_22 =PijFun( [f, b1,b2,bs,b3nl, sigmav1_squre,sigmav2_squre,sigmav3_squre,Dz ],PT,mu, 'auto')
            P_den  = P_00 + mu**2*(2.0*P_01 + P_02 + P_11) + mu**4*(P_03 + P_04 + P_12 + P_13 + 1.0/4.0*P_22)
            P_0den = (2.0*0 + 1.0)  * sp.integrate.simps(P_den,mu,axis=1)
            P_2den = (2.0*2 + 1.0)  * sp.integrate.simps( ((3.*(mu**2)-1.)/2.)*P_den,mu,axis=1)
            P_4den = (2.0*4 + 1.0)  * sp.integrate.simps( ((35.*(mu**4)-30.*(mu**2)+3.)/8. )*P_den,mu,axis=1)  
            P_model.append([P_0den,P_2den,P_4den]) ; PSTYP.append( 'den' ); 
            if((ps_type=='den')and(not do_Multi)):PSRSD=P_den   
        if ('mom' in ps_type):
            P_00,P_01,P_02,P_03,P_04,P_11,P_12,P_13,P_22 =PijFun( [f, b1v,b2v,bsv,b3nlv, sigmavv1_squre,sigmavv2_squre,sigmavv3_squre,Dz ],PT,mu, 'auto')
            P_mom  = ((az*Hz)**2/(PT[0,0:]*PT[0,0:])*(P_11 + mu**2*(2.0*P_12 + 3.0*P_13 + P_22)).T).T
            P_0mom = (2.0*0 + 1.0)  * sp.integrate.simps(P_mom,mu,axis=1)
            P_2mom = (2.0*2 + 1.0)  * sp.integrate.simps( ((3.*(mu**2)-1.)/2.)*P_mom,mu,axis=1) 
            P_4mom = (2.0*4 + 1.0)  * sp.integrate.simps( ((35.*(mu**4)-30.*(mu**2)+3.)/8. )*P_mom,mu,axis=1)  
            P_model.append([P_0mom,P_2mom,P_4mom])  ;  PSTYP.append( 'mom' ); 
            if((ps_type=='mom')and(not do_Multi)):PSRSD=P_mom  
        if ('crs' in ps_type):
            P_00,P_01,P_02,P_03,P_04,P_11,P_12,P_13,P_22     =PijFun( [f, b1,b2,bs,b3nlv, b1v,b2v,bsv,  sigmav1_squre, sigmav2_squre, sigmav3_squre , Dz],PT,mu, 'crs')
            if( 'crsv' in ps_type):
                P_00,P_01,P_02,P_03,P_04,P_11,P_12,P_13,P_22 =PijFun( [f, b1,b2,bs,b3nl , b1v,b2v,bsv,  sigmavv1_squre,sigmavv2_squre,sigmavv3_squre, Dz],PT,mu, 'crs')
            P_crs  = ((az*Hz)/PT[0,0:]*((mu*(P_01 + P_02 + P_11 + mu**2*(3.0/2.0*P_03 + 2.0*P_04 + 3.0/2.0*P_12 + 2.0*P_13  + 1.0/2.0*P_22 ))).T)).T
            P_1crs = (2.0*1 + 1.0)  * sp.integrate.simps(mu*P_crs,mu,axis=1)
            P_3crs = (2.0*3 + 1.0)  * sp.integrate.simps( (1./2.*(5.*mu**3-3.*mu) )*P_crs,mu,axis=1)   
            P_model.append([P_1crs,P_3crs])  ; PSTYP.append( 'crs' ); 
            if((ps_type=='crs')and(not do_Multi)):PSRSD=P_crs
            if((ps_type=='crsv')and(not do_Multi)):PSRSD=P_crs 
    if(GaliTrans):
        P_00,P_01,P_02,P_03,P_04,P_11,P_12,P_13,P_22 =PijFun( [f, b1,b2,bs,b3nl, sigmav1_squre,sigmav2_squre,sigmav3_squre,Dz ],PT,mu, 'auto')
        P_den  = P_00 + mu**2*(2.0*P_01 + P_02 + P_11) + mu**4*(P_03 + P_04 + P_12 + P_13 + 1.0/4.0*P_22)   
        P_00,P_01,P_02,P_03,P_04,P_11,P_12,P_13,P_22 =PijFun( [f, b1v,b2v,bsv,b3nlv, sigmavv1_squre,sigmavv2_squre,sigmavv3_squre,Dz ],PT,mu, 'auto')
        P_mom  = ((az*Hz)**2/(PT[0,0:]*PT[0,0:])*(P_11 + mu**2*(2.0*P_12 + 3.0*P_13 + P_22)).T).T
        P_00,P_01,P_02,P_03,P_04,P_11,P_12,P_13,P_22     =PijFun( [f, b1,b2,bs,b3nlv, b1v,b2v,bsv,  sigmav1_squre, sigmav2_squre, sigmav3_squre , Dz],PT,mu, 'crs')
        if( 'crsv' in ps_type):
            P_00,P_01,P_02,P_03,P_04,P_11,P_12,P_13,P_22 =PijFun( [f, b1,b2,bs,b3nl , b1v,b2v,bsv,  sigmavv1_squre,sigmavv2_squre,sigmavv3_squre, Dz],PT,mu, 'crs')
        P_crs  = ((az*Hz)/PT[0,0:]*((mu*(P_01 + P_02 + P_11 + mu**2*(3.0/2.0*P_03 + 2.0*P_04 + 3.0/2.0*P_12 + 2.0*P_13  + 1.0/2.0*P_22 ))).T)).T
        if ('den' in ps_type):
            P_0den = (2.0*0 + 1.0)  * sp.integrate.simps(P_den,mu,axis=1)
            P_2den = (2.0*2 + 1.0)  * sp.integrate.simps( ((3.*(mu**2)-1.)/2.)*P_den,mu,axis=1)
            P_4den = (2.0*4 + 1.0)  * sp.integrate.simps( ((35.*(mu**4)-30.*(mu**2)+3.)/8. )*P_den,mu,axis=1)  
            P_model.append([P_0den,P_2den,P_4den]) ; PSTYP.append( 'den' ); 
            if((ps_type=='den')and(not do_Multi)):PSRSD=P_den   
        if ('mom' in ps_type):
            P_mom  = P_mom +  epsilon**2 * P_den + epsilon**2/(2.*np.pi)**3
            P_0mom = (2.0*0 + 1.0)  * sp.integrate.simps(P_mom,mu,axis=1)
            P_2mom = (2.0*2 + 1.0)  * sp.integrate.simps( ((3.*(mu**2)-1.)/2.)*P_mom,mu,axis=1) 
            P_4mom = (2.0*4 + 1.0)  * sp.integrate.simps( ((35.*(mu**4)-30.*(mu**2)+3.)/8. )*P_mom,mu,axis=1)  
            P_model.append([P_0mom,P_2mom,P_4mom])  ;  PSTYP.append( 'mom' ); 
            if((ps_type=='mom')and(not do_Multi)):PSRSD=P_mom  
        if ('crs' in ps_type):
            P_crs  = P_crs + epsilon * P_den
            P_1crs = (2.0*1 + 1.0)  * sp.integrate.simps(mu*P_crs,mu,axis=1)
            P_3crs = (2.0*3 + 1.0)  * sp.integrate.simps( (1./2.*(5.*mu**3-3.*mu) )*P_crs,mu,axis=1)   
            P_model.append([P_1crs,P_3crs])  ; PSTYP.append( 'crs' ); 
            if((ps_type=='crs')and(not do_Multi)):PSRSD=P_crs
            if((ps_type=='crsv')and(not do_Multi)):PSRSD=P_crs 
    if(do_Multi):
        return P_model,PSTYP
    else:
        return PT[0,0:],mu, PSRSD 
def Pkmod_Kaiser_Fun(parm,Sig8_fid,k,PL,ps_type, az,Hz, do_Multi=True, GaliTrans=False  ):        
    if(GaliTrans):
        fsigma8,bsigma8,sigvd,sigvv,epsilon=parm
    else:    
        fsigma8,bsigma8,sigvd,sigvv=parm
    # model PS:
    P_model=[];PSTYP=[];PSRSD=[] 
    mu = np.linspace(0.0, 1.0, 300) 
    if(not GaliTrans):
        if ('den' in ps_type):
            P_den=np.zeros((len(k),len(mu)))  
            for ss in range(len(mu)):
                u=mu[ss]
                P_den[:,ss]=( bsigma8/Sig8_fid + fsigma8/Sig8_fid * u**2 )*( bsigma8/Sig8_fid + fsigma8/Sig8_fid * u**2 ) * (1.+0.5*(k*u*sigvd)**2)**(-1./2.)*(1.+0.5*(k*u*sigvd)**2)**(-1./2.) * PL   
            P_0den = (2.0*0 + 1.0)  * sp.integrate.simps(P_den,mu,axis=1)
            P_2den = (2.0*2 + 1.0)  * sp.integrate.simps( ((3.*(mu**2)-1.)/2.)*P_den,mu,axis=1)
            P_4den = (2.0*4 + 1.0)  * sp.integrate.simps( ((35.*(mu**4)-30.*(mu**2)+3.)/8. )*P_den,mu,axis=1)  
            P_model.append([P_0den,P_2den,P_4den]) ; PSTYP.append( 'den' ); 
            if((ps_type=='den')and(not do_Multi)):PSRSD=P_den   
        if ('mom' in ps_type):
            P_mom=np.zeros((len(k),len(mu)))  
            for ss in range(len(mu)):
                u=mu[ss]  
                P_mom[:,ss]=( az * Hz *fsigma8/Sig8_fid * u / k )        *( az * Hz *fsigma8/Sig8_fid * u / k )         * np.sinc(k*sigvv)                 *np.sinc(k*sigvv)                  * PL
            P_0mom = (2.0*0 + 1.0)  * sp.integrate.simps(P_mom,mu,axis=1)
            P_2mom = (2.0*2 + 1.0)  * sp.integrate.simps( ((3.*(mu**2)-1.)/2.)*P_mom,mu,axis=1) 
            P_4mom = (2.0*4 + 1.0)  * sp.integrate.simps( ((35.*(mu**4)-30.*(mu**2)+3.)/8. )*P_mom,mu,axis=1)  
            P_model.append([P_0mom,P_2mom,P_4mom])  ;  PSTYP.append( 'mom' ); 
            if((ps_type=='mom')and(not do_Multi)):PSRSD=P_mom  
        if ('crs' in ps_type):
            P_crs=np.zeros((len(k),len(mu)))  
            for ss in range(len(mu)):
                u=mu[ss]  
                P_crs[:,ss]=( bsigma8/Sig8_fid + fsigma8/Sig8_fid * u**2 )*( az * Hz *fsigma8/Sig8_fid * u / k )         * (1.+0.5*(k*u*sigvd)**2)**(-1./2.)*np.sinc(k*sigvv)                  * PL    
            P_1crs = (2.0*1 + 1.0)  * sp.integrate.simps(mu*P_crs,mu,axis=1)
            P_3crs = (2.0*3 + 1.0)  * sp.integrate.simps( (1./2.*(5.*mu**3-3.*mu) )*P_crs,mu,axis=1)   
            P_model.append([P_1crs,P_3crs])  ; PSTYP.append( 'crs' ); 
            if((ps_type=='crs')and(not do_Multi)):PSRSD=P_crs  
    if(GaliTrans):
        P_den=np.zeros((len(k),len(mu)))  
        for ss in range(len(mu)):
            u=mu[ss]
            P_den[:,ss]=( bsigma8/Sig8_fid + fsigma8/Sig8_fid * u**2 )*( bsigma8/Sig8_fid + fsigma8/Sig8_fid * u**2 ) * (1.+0.5*(k*u*sigvd)**2)**(-1./2.)*(1.+0.5*(k*u*sigvd)**2)**(-1./2.) * PL             
        P_mom=np.zeros((len(k),len(mu)))  
        for ss in range(len(mu)):
            u=mu[ss]  
            P_mom[:,ss]=( az * Hz *fsigma8/Sig8_fid * u / k )        *( az * Hz *fsigma8/Sig8_fid * u / k )         * np.sinc(k*sigvv)                 *np.sinc(k*sigvv)                  * PL        
        P_crs=np.zeros((len(k),len(mu)))  
        for ss in range(len(mu)):
            u=mu[ss]  
            P_crs[:,ss]=( bsigma8/Sig8_fid + fsigma8/Sig8_fid * u**2 )*( az * Hz *fsigma8/Sig8_fid * u / k )         * (1.+0.5*(k*u*sigvd)**2)**(-1./2.)*np.sinc(k*sigvv)                  * PL    
        if ('den' in ps_type):
            P_0den = (2.0*0 + 1.0)  * sp.integrate.simps(P_den,mu,axis=1)
            P_2den = (2.0*2 + 1.0)  * sp.integrate.simps( ((3.*(mu**2)-1.)/2.)*P_den,mu,axis=1)
            P_4den = (2.0*4 + 1.0)  * sp.integrate.simps( ((35.*(mu**4)-30.*(mu**2)+3.)/8. )*P_den,mu,axis=1)  
            P_model.append([P_0den,P_2den,P_4den]) ; PSTYP.append( 'den' ); 
            if((ps_type=='den')and(not do_Multi)):PSRSD=P_den   
        if ('mom' in ps_type):
            P_mom  = P_mom +  epsilon**2 * P_den + epsilon**2/(2.*np.pi)**3
            P_0mom = (2.0*0 + 1.0)  * sp.integrate.simps(P_mom,mu,axis=1)
            P_2mom = (2.0*2 + 1.0)  * sp.integrate.simps( ((3.*(mu**2)-1.)/2.)*P_mom,mu,axis=1) 
            P_4mom = (2.0*4 + 1.0)  * sp.integrate.simps( ((35.*(mu**4)-30.*(mu**2)+3.)/8. )*P_mom,mu,axis=1)  
            P_model.append([P_0mom,P_2mom,P_4mom])  ;  PSTYP.append( 'mom' ); 
            if((ps_type=='mom')and(not do_Multi)):PSRSD=P_mom  
        if ('crs' in ps_type):
            P_crs  = P_crs + epsilon * P_den
            P_1crs = (2.0*1 + 1.0)  * sp.integrate.simps(mu*P_crs,mu,axis=1)
            P_3crs = (2.0*3 + 1.0)  * sp.integrate.simps( (1./2.*(5.*mu**3-3.*mu) )*P_crs,mu,axis=1)   
            P_model.append([P_1crs,P_3crs])  ; PSTYP.append( 'crs' ); 
            if((ps_type=='crs')and(not do_Multi)):PSRSD=P_crs               
    if(do_Multi):
        return P_model,PSTYP
    else:
        return k,mu,PSRSD  
    
# 3. calculate window function convolved PS:-----------------------------------
def PkmodMulti_Fun(params,OnlyL0,Sig8_fid, k_obs, kmod_conv,WF_Kobs_Kmod,WF_rand,PT,ps_type, az,Hz,Dz, WFCon_type,model_typt='nonlin',GaliTrans=False):
    if((WFCon_type!='Ross')and(WFCon_type!='CosmPS')and(WFCon_type!='Pypower')):
        kobs = k_obs+0.-0.
        k_obs   = np.concatenate(np.array([list(kobs),list(kobs),list(kobs)]))
        kmod_conv  = np.array([list(kobs),list(kobs),list(kobs)])
        WF_Kobs_Kmod  = [np.eye(3*len(kobs)),np.eye(3*len(kobs)),np.eye(2*len(kobs))] 
        WFCon_type ='CosmPS'
    if(model_typt=='lin'):Pk_mods , psty= Pkmod_Kaiser_Fun(params,Sig8_fid,PT[0,0:],PT[1,0:],ps_type, az,Hz   , True, GaliTrans  )    
    else:                 Pk_mods , psty= Pkmod_Fun(       params,Sig8_fid,PT,               ps_type, az,Hz,Dz, True, GaliTrans )
    Pk_mod   = []   ;  Pk_modunc   = []   ; kmod_uncon=[]
    for i in range(len(Pk_mods)):
        Pk_modss=np.concatenate(Pk_mods[i] )
        if(WFCon_type=='Ross'):
            Nic=3
            Nks=len(k_obs)
            if(len(Pk_mods)==2):Nks=Nks//2
            if(i==0):
                P_spline_0 = sp.interpolate.splrep(PT[0,:],Pk_modss[0:(len(Pk_modss)//Nic)],s=0)
                Pk_mod0    = sp.interpolate.splev(kmod_conv , P_spline_0)
                Pk_mod     = Pk_CONV_ross(k_obs[:Nks] ,WF_Kobs_Kmod,Pk_mod0,WF_rand[0],WF_rand[1]) 
            if(i==1): 
                P_spline_0 = sp.interpolate.splrep(PT[0,:],Pk_modss[0:(len(Pk_modss)//Nic)],s=0)
                Pk_modp0   = sp.interpolate.splev(kmod_conv , P_spline_0)
                Pk_modp    = Pk_CONV_ross(k_obs[:Nks] ,WF_Kobs_Kmod,Pk_modp0,WF_rand[0],WF_rand[1])
                Pk_mod     = np.concatenate((Pk_mod,Pk_modp))
                Pk_mod0    = np.concatenate((Pk_mod0,Pk_modp0))
                kmod_conv  = np.concatenate((kmod_conv,kmod_conv))
            if(len(Pk_mods)==1):  
                return Pk_mod, k_obs, Pk_mod0, kmod_conv
            else:
                if(i==1):
                    return Pk_mod, k_obs, Pk_mod0, kmod_conv
        if(WFCon_type!='Ross')and(OnlyL0):    
            Nic=3
            Nks=len(k_obs)
            if(i==0):
                P_spline_0 = sp.interpolate.splrep(PT[0,:],Pk_modss[0:(len(Pk_modss)//Nic)],s=0)
                Pk_mod0    = sp.interpolate.splev(kmod_conv[i] , P_spline_0) 
                Pk_mod     = Pk_CONV_multi( WF_Kobs_Kmod[i],[Pk_mod0,Pk_mod0,Pk_mod0],  psty[i])[0]
            if(i==1): 
                P_spline_0 = sp.interpolate.splrep(PT[0,:],Pk_modss[0:(len(Pk_modss)//Nic)],s=0)
                Pk_modp0   = sp.interpolate.splev(kmod_conv[i] , P_spline_0) 
                Pk_modp    = Pk_CONV_multi( WF_Kobs_Kmod[i],[Pk_modp0,Pk_modp0,Pk_modp0],  psty[i])[0]
                Pk_mod     = np.concatenate((Pk_mod,Pk_modp))
                Pk_mod0    = np.concatenate((Pk_mod0,Pk_modp0))
                kmod_con   = np.concatenate((kmod_conv[i],kmod_conv[i] ))
            if(len(Pk_mods)==1):  
                return Pk_mod, k_obs, Pk_mod0, kmod_conv[0]
            else:
                if(i==1):
                    return Pk_mod, k_obs, Pk_mod0, kmod_con 
        if(WFCon_type!='Ross')and(not OnlyL0): 
            if(psty[i]=='den')or(psty[i]=='mom'):
                Nic=3
            if(psty[i]=='crs'):
                Nic=2
            P_spline_0     = sp.interpolate.splrep(PT[0,:],Pk_modss[0                     :(len(Pk_modss)//Nic)  ],s=0)
            Pk_mod0        = sp.interpolate.splev(kmod_conv[i], P_spline_0)
            P_spline_2     = sp.interpolate.splrep(PT[0,:],Pk_modss[(len(Pk_modss)//Nic)  :(len(Pk_modss)//Nic*2)],s=0)
            Pk_mod2        = sp.interpolate.splev(kmod_conv[i], P_spline_2)            
            if(psty[i]=='den')or(psty[i]=='mom'):
                P_spline_4 = sp.interpolate.splrep(PT[0,:],Pk_modss[(len(Pk_modss)//Nic*2):(len(Pk_modss)//Nic*3)],s=0)
                Pk_mod4    = sp.interpolate.splev(kmod_conv[i], P_spline_4) 
            if(psty[i]=='den'):
                Pk_modm0 ,Pk_modm2 ,Pk_modm4    = Pk_CONV_multi( WF_Kobs_Kmod[0],[Pk_mod0,Pk_mod2,Pk_mod4],  psty[i])
                Pk_mod.append(Pk_modm0)  ; Pk_mod.append(Pk_modm2)        ; Pk_mod.append(Pk_modm4) 
                Pk_modunc.append(Pk_mod0); Pk_modunc.append(Pk_mod2); Pk_modunc.append(Pk_mod4)
                kmod_uncon.append(kmod_conv[0]);kmod_uncon.append(kmod_conv[0]);kmod_uncon.append(kmod_conv[0])
            if(psty[i]=='mom'):
                Pk_modm0 ,Pk_modm2 ,Pk_modm4    = Pk_CONV_multi( WF_Kobs_Kmod[1],[Pk_mod0,Pk_mod2,Pk_mod4],  psty[i])
                Pk_mod.append(Pk_modm0)  ; Pk_mod.append(Pk_modm2)        ; Pk_mod.append(Pk_modm4) 
                Pk_modunc.append(Pk_mod0); Pk_modunc.append(Pk_mod2); Pk_modunc.append(Pk_mod4)
                kmod_uncon.append(kmod_conv[1]);kmod_uncon.append(kmod_conv[1]);kmod_uncon.append(kmod_conv[1])
            if(psty[i]=='crs'):
                Pk_modm1 ,Pk_modm3      = Pk_CONV_multi( WF_Kobs_Kmod[2],[Pk_mod0,Pk_mod2 ],  psty[i])
                Pk_mod.append(Pk_modm1)  ; Pk_mod.append(Pk_modm3) 
                Pk_modunc.append(Pk_mod0); Pk_modunc.append(Pk_mod2)
                kmod_uncon.append(kmod_conv[2]);kmod_uncon.append(kmod_conv[2])
    if(WFCon_type!='Ross')and(not OnlyL0):
        return   np.concatenate(Pk_mod),   k_obs,     np.concatenate(Pk_modunc),  np.concatenate(kmod_uncon)    
                
# 4. pik up used multipoles: --------------------------------------------------
def PSmulti_pickup(ind_fit,ps_type,Pk_obs,Pk_modc):
    Ps_type = ps_type.split(' ')
    mt=[]
    for i in range(len(Ps_type) ):
        ps_type0=Ps_type[i].split('-')
        tmp=[]
        for j in range(len(ps_type0[1])):
            tmp.append( int(ps_type0[1][j]) )
        mt.append(np.array(tmp))  
    if('den' in ps_type)and('mom' not in ps_type)and('crs' not in ps_type): nn=3               
    if('mom' in ps_type)and('den' not in ps_type)and('crs' not in ps_type): nn=3
    if('crs' in ps_type)and('den' not in ps_type)and('den' not in ps_type): nn=2    
    if('den' in ps_type)and('mom' in ps_type)and('crs' not in ps_type): nn=6
    if('den' in ps_type)and('mom' in ps_type)and('crs'     in ps_type): nn=8
    if('den' in ps_type)and('mom' not in ps_type)and('crs' in ps_type): nn=5
    if('den' not in ps_type)and('mom' in ps_type)and('crs' in ps_type): nn=5                    
    Nf=len(ind_fit)//nn
    Pk_obs=Pk_obs[ind_fit]
    Pk_modc=Pk_modc[ind_fit]
    Pko=[]; Pkm=[]  ; I=0;J=1
    for i in range(len(mt)):  
      if(i==0): 
        if(0 in mt[i])or(1 in mt[i]):
            Pko.append(Pk_obs[    I*Nf:J*Nf]) 
            Pkm.append(Pk_modc[   I*Nf:J*Nf])
        I=I+1;J=J+1
        if(2 in mt[i])or(3 in mt[i]):
            Pko.append(Pk_obs[    I*Nf:J*Nf])
            Pkm.append(Pk_modc[   I*Nf:J*Nf])
        I=I+1;J=J+1
        if(4 in mt[i]):     
            Pko.append(Pk_obs[    I*Nf:J*Nf])
            Pkm.append(Pk_modc[   I*Nf:J*Nf]) 
        I=I+1;J=J+1     
      if(i==1):    
        if(0 in mt[i])or(1 in mt[i]):
            Pko.append(Pk_obs[    I*Nf:J*Nf]) 
            Pkm.append(Pk_modc[   I*Nf:J*Nf])
        I=I+1;J=J+1
        if(2 in mt[i])or(3 in mt[i]):
            Pko.append(Pk_obs[    I*Nf:J*Nf])
            Pkm.append(Pk_modc[   I*Nf:J*Nf])
        I=I+1;J=J+1
        if(4 in mt[i]):     
            Pko.append(Pk_obs[    I*Nf:J*Nf])
            Pkm.append(Pk_modc[   I*Nf:J*Nf])
        I=I+1;J=J+1                 
      if(i==2):     
        if(1 in mt[i]):
            Pko.append(Pk_obs[    I*Nf:J*Nf]) 
            Pkm.append(Pk_modc[   I*Nf:J*Nf])
        I=I+1;J=J+1
        if(3 in mt[i]):
            Pko.append(Pk_obs[    I*Nf:J*Nf])
            Pkm.append(Pk_modc[   I*Nf:J*Nf])
        I=I+1;J=J+1
    Pk_obs=np.concatenate(Pko)
    Pk_modc=np.concatenate(Pkm)
    return Pk_obs,Pk_modc

#########################     The end of Sec 2.    ############################# 
###############################################################################
        
        
        
        
        
        
        
        















###############################################################################
#####                                                                     #####
#####            Sec 3. The window function convolution matrix            #####
#####                                                                     #####
###############################################################################        
# 1: cosmological functions:
# Speed of light in km/s
LightSpeed = 299792.458
# Calculate distance:
# Calculates H(z)/H0
def Ez(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap):
    fz = ((1.0+redshift)**(3*(1.0+w0+wa*ap)))*np.exp(-3*wa*(redshift/(1.0+redshift)))
    omega_k = 1.0-omega_m-omega_lambda-omega_rad
    return np.sqrt(omega_rad*(1.0+redshift)**4+omega_m*(1.0+redshift)**3+omega_k*(1.0+redshift)**2+omega_lambda*fz)
# The Comoving Distance Integrand
def DistDcIntegrand(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap):
    return 1.0/Ez(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap)
# The Comoving Distance in Mpc
def DistDc(redshift, omega_m, omega_lambda, omega_rad, Hubble_Constant, w0, wa, ap):
    return (LightSpeed/Hubble_Constant)*integrate.quad(DistDcIntegrand, 0.0, redshift, args=(omega_m, omega_lambda, omega_rad, w0, wa, ap))[0]
def spline_Dis2Rsf(OmegaM,OmegaA,Hub,nbin=10000,redmax=2.5):
    dist = np.empty(nbin);red = np.empty(nbin)
    for j in range(nbin):
        red[j] = j*redmax/nbin
        dist[j] = DistDc(red[j],OmegaM,OmegaA, 0.0,Hub,-1.0, 0.0, 0.0)
    rsf_spline = sp.interpolate.splrep(dist,red,  s=0)
    return rsf_spline
def spline_Rsf2Dis(OmegaM,OmegaA,Hub,nbin=10000,redmax=2.5):
    dist = np.empty(nbin);red = np.empty(nbin)
    for j in range(nbin):
        red[j] = j*redmax/nbin
        dist[j] = DistDc(red[j],OmegaM,OmegaA, 0.0,Hub,-1.0, 0.0, 0.0)
    dist_spline = sp.interpolate.splrep(red, dist, s=0)
    return dist_spline
def DisRsfConvert(xdt,types,OmegaM,OmegaA,Hub,nbin=10000,redmax=2.5):
    if(types=='z2d'):
        spl_fun=spline_Rsf2Dis(OmegaM,OmegaA,Hub,nbin,redmax)
        Distc        = splev(xdt, spl_fun)
        return Distc
    if(types=='d2z'):
        spl_fun=spline_Dis2Rsf(OmegaM,OmegaA,Hub,nbin,redmax)
        RSFs        = splev(xdt, spl_fun)
        return RSFs
def Sky2Cat(ra,dec,rsft,OmegaM , OmegaA ,Hub,nbin=1000,redmax=2.):
    disz= DisRsfConvert(rsft,'z2d',OmegaM,OmegaA,Hub,nbin,redmax)
    X = disz*np.cos(dec/180.*np.pi)*np.cos(ra/180.*np.pi)
    Y = disz*np.cos(dec/180.*np.pi)*np.sin(ra/180.*np.pi)
    Z = disz*np.sin(dec/180.*np.pi)    
    return X,Y,Z 
# The Linear Growth Factor Integrand assuming GR
def GrowthFactorGRIntegrand(scale_factor, omega_m, omega_lambda, omega_rad, w0, wa, ap):
    redshift = (1.0/scale_factor)-1.0
    return 1.0/(scale_factor*Ez(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap))**3
# The Linear Growth Factor assuming GR
def GrowthFactorGR(redshift, omega_m, omega_lambda, omega_rad,   w0, wa, ap):
    prefac = Ez(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap)
    scale_factor = 1.0/(1.0+redshift)
    return prefac*integrate.quad(GrowthFactorGRIntegrand, 0.0, scale_factor, args=(omega_m, omega_lambda, omega_rad, w0, wa, ap))[0]
# Omega_M at a given redshift
def Omega_m_z(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap):
    return (omega_m*(1.0+redshift)**3)/(Ez(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap)**2)
# The Linear Growth Factor Integrand for an arbitrary value of gamma
def GrowthFactorGammaIntegrand(scale_factor, gamma, omega_m, omega_lambda, omega_rad, w0, wa, ap):
    redshift = (1.0/scale_factor)-1.0
    return (Omega_m_z(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap)**gamma)/scale_factor
# The Linear Growth Factor for an arbitrary value of gamma
def GrowthFactorGamma(gamma, redshift_low, redshift_high, omega_m, omega_lambda, omega_rad,   w0, wa, ap):
    scale_factor_low = 1.0/(1.0+redshift_low)
    scale_factor_high = 1.0/(1.0+redshift_high)
    return np.exp(integrate.quad(GrowthFactorGammaIntegrand, scale_factor_low, scale_factor_high, args=(gamma, omega_m, omega_lambda, omega_rad, w0, wa, ap))[0])
def Fsigma8_Fun(gamma,RsFeff,Sig8_fid, OmegaM, OmegaA ):
    sig8_fac  = GrowthFactorGR(1000., OmegaM, OmegaA, 0.0,   -1.0, 0.0, 0.0)/GrowthFactorGR(0.0, OmegaM, OmegaA, 0.0,   -1.0, 0.0, 0.0)
    gamma_fac = GrowthFactorGamma(gamma, 1000.,  RsFeff ,  OmegaM, OmegaA, 0.0,   -1.0, 0.0, 0.0)
    OmegaMz   = Omega_m_z(RsFeff, OmegaM, OmegaA, 0.0, -1.0, 0.0, 0.0)
    fsig8     = OmegaMz**gamma*Sig8_fid*sig8_fac*gamma_fac 
    return fsig8 
def Dz_Fun(RsFeff, OmegaM, OmegaA ):
    Dz = GrowthFactorGR(RsFeff, OmegaM, OmegaA, 0.0,   -1.0, 0.0, 0.0)/GrowthFactorGR(0.0, OmegaM, OmegaA, 0.0,   -1.0, 0.0, 0.0)
    return Dz
def logd_to_Vp(Logd,Rsf,OmegaM):
    deccel = 3.0*OmegaM/2.0 - 1.0
    Vmod   = Rsf*LightSpeed*(1.0 + 0.5*(1.0 - deccel)*Rsf - (2.0 - deccel - 3.0*deccel*deccel)*Rsf*Rsf/6.0)
    vpec   = np.log(10.)*Vmod/(1.0+Vmod/LightSpeed) * Logd
    return vpec       
def VpRand_Fun(logd,cz,czR,NczBin,OmegaM):
    pv=logd_to_Vp(logd,cz/LightSpeed,OmegaM)
    pvR=[];ev=[]
    Num,Bin=np.histogram(cz,NczBin);
    for i in range(len(Num)+1):
        if(i<=len(Num)-1):
            ind=(cz>=Bin[i])&(cz<=Bin[i+1])
            er=np.std(pv[ind])
            ev.append(er)
    ev=np.array(ev)
    x=Bin[:-1]+0.5*np.diff(Bin)
    k,b=np.polyfit(x,ev,deg=1)
    epvR=k*czR+b
    np.random.seed(326)
    pvR=np.random.normal(loc=0.0, scale=epvR, size=len(epvR))
    return pvR, epvR

# 2. grid the data:------------------------------------------------------------
def discret_rand(xpos,ypos,zpos,nx,ny,nz,lx,ly,lz,   wei=1.):
    datgrid,edges = np.histogramdd(np.vstack([xpos,ypos,zpos]).transpose(),bins=(nx,ny,nz),range=((-lx/2.,lx/2.),(-ly/2.,ly/2.),(-lz/2.,lz/2.)),weights=wei)
    return datgrid 
def Prep_winFUN(nrand,  fkp,rand_x,rand_y,rand_z,epvR,wts,sigv,nx,ny,nz,lx,ly,lz,types ):
    vol = lx*ly*lz
    randgrid = discret_rand(rand_x,rand_y,rand_z,nx,ny,nz,lx,ly,lz,  np.ones(len(rand_x))*wts )
    wingrid = randgrid
    if(types=='den'):  
        weigrid=np.zeros((nx,ny,nz))
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    weigrid[i,j,k] = 1./(1.0+(randgrid[i,j,k] *float(nx*ny*nz)/vol)*fkp) 
                    if(randgrid[i,j,k]==0): weigrid[i,j,k]=0                            
    if(types=='mom'):
        pverrgrid = discret_rand(rand_x,rand_y,rand_z,nx,ny,nz,lx,ly,lz, np.ones(len(rand_x))*epvR*wts)
        pverrgrid[randgrid > 0] /= randgrid[randgrid > 0]
        pverrgrid += sigv**2
        weigrid=np.zeros((nx,ny,nz))
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if(fkp==0.):
                        weigrid[i,j,k] = 1.
                    else:
                        weigrid[i,j,k] = 1./(pverrgrid[i,j,k]+(randgrid[i,j,k] *float(nx*ny*nz)/vol)*fkp)
                    if(randgrid[i,j,k]==0.): weigrid[i,j,k]=0
    return wingrid,weigrid
    
    
# 2 set up k bin functions:----------------------------------------------------
def getkspec(nx,ny,nz,lx,ly,lz):
    kx = 2.*np.pi*np.fft.fftfreq(nx,d=lx/nx)
    ky = 2.*np.pi*np.fft.fftfreq(ny,d=ly/ny)
    kz = 2.*np.pi*np.fft.fftfreq(nz,d=lz/nz)
    indep = np.full((nx,ny,nz),True,dtype=bool)
    indep[0,0,0] = False
    kspec = np.sqrt(kx[:,np.newaxis,np.newaxis]**2 + ky[np.newaxis,:,np.newaxis]**2 + kz[np.newaxis,np.newaxis,:]**2)
    kspec[0,0,0] = 1.
    muspec = np.absolute(kx[:,np.newaxis,np.newaxis])/kspec
    kspec[0,0,0] = 0.
    return kspec,muspec,indep  

def binpk(pkspec,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin):
    #print 'Binning in angle-averaged bins...'
    kspec,muspec,indep = getkspec(nx,ny,nz,lx,ly,lz)
    pkspec = pkspec[indep == True]
    kspec = kspec[indep == True]
    ikbin = np.digitize(kspec,np.linspace(kmin,kmax,nkbin+1))
    nmodes,pk = np.zeros(nkbin,dtype=int),np.full(nkbin,-1.)
    for ik in range(nkbin):
      nmodes[ik] = int(np.sum(np.array([ikbin == ik+1])))
      if (nmodes[ik] > 0):
        pk[ik] = np.mean(pkspec[ikbin == ik+1])
      if(ik==0)and(pk[ik]==-1.):pk[ik]=0.# the first kbin is not used. 
    return pk,nmodes 

# 3. CosmPS convolution matrix:-------------------------------------------------
def ConvMat_CosmPS(kmin,kmax,nkbin,kmaxc,nkbinc,nx,ny,nz,lx,ly,lz,w,Win,w2,Win2,PStype,kmod_scale='log'):
    # Initializations
    if(PStype!='crs'):nlconv = 3 # Number of multipoles to include in convolution
    if(PStype=='crs'):nlconv = 2
    if(kmod_scale=='lin'):
        dkc = (kmaxc-kmin)/nkbinc
        k1 = np.linspace(kmin,kmaxc-dkc,nkbinc)
        k2 = np.linspace(kmin+dkc,kmaxc,nkbinc)
    #if(kmod_scale=='geo'):    
    #    spc= np.geomspace(1e-8,kmaxc,nkbinc+1);
    #    k1 = spc[:-1]; k2=spc[1:] ; k1[0]=kmin
    if(kmod_scale=='log'):
        spc=(1.111*kmaxc*(np.logspace(0., 1., nkbinc+1 )-1.)/10.)  
        spc[-1]=kmaxc; spc[0]=0. ;spc=np.sort(kmaxc-spc)
        k1=spc[:-1]; k2=spc[1:]     
    convmat = np.zeros((nlconv*(nkbin-1),nlconv*nkbinc))
    # Convolve series of unit vectors
    iconv = -1
    for imult in range(nlconv):
      for ik in range(nkbinc):
        iconv += 1
        print( 'Obtaining convolution for bin',iconv+1,'of',nlconv*nkbinc,'...')
        pk0,pk1,pk2, pk3,pk4 = 0.,0.,0.,0.,0.
        if(PStype!='crs'):
            if (imult == 0):   pk0 = 1.
            elif (imult == 1): pk2 = 1.          
            elif (imult == 2): pk4 = 1.
            kmin1,kmax1 = k1[ik],k2[ik]
            pk=[pk0, pk2,  pk4]
            pk0con, pk2con, pk4con = Conv_Fun(nx,ny,nz,lx,ly,lz,w,Win,w2,Win2,kmin,kmax,nkbin,kmin1,kmax1,pk,PStype)
            convmat[:,iconv] = np.concatenate((pk0con, pk2con, pk4con))
        if(PStype=='crs'): 
            if (imult == 0):   pk1 = 1.
            elif (imult == 1): pk3 = 1.          
            kmin1,kmax1 = k1[ik],k2[ik]
            pk=[pk1,  pk3]
            pk1con, pk3con = Conv_Fun(nx,ny,nz,lx,ly,lz,w,Win,w2,Win2,kmin,kmax,nkbin,kmin1,kmax1,pk,PStype)
            convmat[:,iconv] = np.concatenate((pk1con, pk3con))            
    kd=np.linspace(kmin,kmax,nkbin+1)[:-1]
    kc=np.concatenate((k1,np.array([k2[-1]])))
    return convmat,kd[:-1]+0.5*np.diff(kd),kc[:-1]+0.5*np.diff(kc)
# https://arxiv.org/pdf/1801.04969.pdf 
def Conv_Fun(nx,ny,nz,lx,ly,lz,w,Win,w2,Win2,kmin,kmax,nkbin,kmin1,kmax1,pk,PStype ):
    if(PStype!='crs'):
        pk0, pk2, pk4=pk
        nlmod = 3 
        nl=3
        uselp = np.full(nlmod,True,dtype=bool)
        if (pk0 == 0.): uselp[0] = False
        if (pk2 == 0.): uselp[1] = False
        if (pk4 == 0.): uselp[2] = False 
    if(PStype=='crs'): 
        pk1, pk3=pk
        nlmod = 2 
        nl=2
        uselp = np.full(nlmod,True,dtype=bool)
        if (pk1 == 0.): uselp[0] = False
        if (pk3 == 0.): uselp[1] = False
    # grid cells' possition and wave numbers. 
    dx,dy,dz = lx/nx,ly/ny,lz/nz
    x  = dx*np.arange(nx) - lx/2. + 0.5*dx
    y  = dy*np.arange(ny) - ly/2. + 0.5*dy
    z  = dz*np.arange(nz) - lz/2. + 0.5*dz
    kx = 2.*np.pi*np.fft.fftfreq(nx,d=dx)
    ky = 2.*np.pi*np.fft.fftfreq(ny,d=dy)
    kz = 2.*np.pi*np.fft.fftfreq(nz,d=dz)
    # Obtain spherical polar angles over the grid
    rgrid  = np.sqrt(x[:,np.newaxis,np.newaxis]**2 + y[np.newaxis,:,np.newaxis]**2 + z[np.newaxis,np.newaxis,:]**2)
    rtheta = np.arctan2(z[np.newaxis,np.newaxis,:],y[np.newaxis,:,np.newaxis])
    rphi   = np.where(rgrid>0.,np.arccos(x[:,np.newaxis,np.newaxis]/rgrid),0.)
    kgrid  = np.sqrt(kx[:,np.newaxis,np.newaxis]**2 + ky[np.newaxis,:,np.newaxis]**2 + kz[np.newaxis,np.newaxis,:]**2)
    ktheta = np.arctan2(kz[np.newaxis,np.newaxis,:],ky[np.newaxis,:,np.newaxis])
    kphi   = np.where(kgrid>0.,np.arccos(kx[:,np.newaxis,np.newaxis]/kgrid),0.)     
    mask   = (kgrid >= kmin1) & (kgrid < kmax1)
    # normalization factor:#
    nw  = w*Win/np.sum(Win) 
    nwb = np.fft.fftn( nw )
    Nc  = nx*ny*nz
    I   = Nc*np.sum(nw**2)  
    nw2 = w2*Win2/np.sum(Win2) 
    nwb2= np.fft.fftn( nw2 )
    I2  = Nc*np.sum(nw2**2) 
    # calculate convolution:
    pk0con,pk1con,pk2con,pk3con,pk4con = np.zeros(nkbin-1),np.zeros(nkbin-1),np.zeros(nkbin-1),np.zeros(nkbin-1),np.zeros(nkbin-1)
    for il in range(nl): 
        if(PStype!='crs'): l = 2*il
        if(PStype=='crs'): l = 2*(il+1)-1
        pkcon = np.zeros((nx,ny,nz))
        for m in range(-l,l+1):
            Ylm_r=sph_harm(m,l,rtheta ,rphi)
            Ylm_k=sph_harm(m,l,ktheta ,kphi) 
            for ilp in range(nlmod):
              if (uselp[ilp]):      
                if(PStype!='crs'):lp = 2*ilp
                if(PStype=='crs'):lp = 2*(ilp+1)-1               
                norm = (4.*np.pi)**2/(2.*lp+1.) /I   
                norm2= (4.*np.pi)**2/(2.*lp+1.) /I2 
                Pl = np.zeros((nx,ny,nz))
                if(PStype!='crs'):
                    if(ilp == 0):   Pl[mask] = pk0
                    elif(ilp == 1): Pl[mask] = pk2
                    elif(ilp == 2): Pl[mask] = pk4 
                if(PStype=='crs'):
                    if(ilp == 0):   Pl[mask] = pk1
                    elif(ilp == 1): Pl[mask] = pk3    
                for mpr in range(-lp,lp+1):
                    Ylm_kp = sph_harm(mpr,lp,ktheta ,kphi)
                    Ylm_rp = sph_harm(mpr,lp,rtheta ,rphi)
                    Slmlm  = np.fft.fftn( nw *Ylm_r*np.conj(Ylm_rp) )
                    Slmlm2 = np.fft.fftn( nw2*Ylm_r*np.conj(Ylm_rp) )
                    PY     = np.fft.fftn( np.conj(Ylm_kp) * Pl )
                    nS     = np.fft.fftn(nwb2 * np.conj(Slmlm))
                    nS2    = np.fft.fftn(nwb * np.conj(Slmlm2))
                    pkcon  = pkcon + np.sqrt(norm*norm2) * np.real( Ylm_k * np.fft.ifftn( PY* 0.5*(nS+nS2) ) )  
        pkc ,nmodes = binpk(pkcon,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
        if(PStype!='crs'):
            if(il == 0):   pk0con = pkc[:-1]  
            elif(il == 1): pk2con = pkc[:-1]   
            elif(il == 2): pk4con = pkc[:-1]             
        if(PStype=='crs'): 
            if(il == 0):   pk1con = pkc[:-1]    
            elif(il == 1): pk3con = pkc[:-1]             
    if(PStype!='crs'):
        return   pk0con, pk2con, pk4con 
    if(PStype=='crs'):
        return   pk1con, pk3con 

  
# 4. Ross convolution method:---------------------------------------------------   
def PS_extender(k,Pk,k_start,k_end):
    #k_store=k
    XX=k[k_start:k_end]
    YY=Pk[k_start:k_end]
    x=np.log10(XX)
    y=np.log10(YY)
    KK,BB=np.polyfit(x,y,1)
    dks=0.001
    krs=np.zeros(int((10.0-max(k))/dks))
    for i in range(int((10.0-max(k))/dks)):
        krs[i]=max(k)+(i+1)*dks
    KRAND=k
    PKRAND=Pk
    k=np.zeros(len(KRAND)+int((10.0-max(k))/dks))
    Pk=np.zeros(len(k))
    k[0:len(KRAND)]=KRAND
    k[len(KRAND):len(KRAND)+int((10.0-max(k))/dks)]=krs
    Pk[0:len(KRAND)]=PKRAND
    Pk[len(KRAND):len(k)]=10.0**(KK*np.log10(krs)+BB)
    return k,Pk

def ConvMat_Ross(Ki,kjmin,kjmax,nkbinc,WF_rand,Ncosa,epsilon_par):  
    Kj=np.linspace(kjmin,kjmax,nkbinc+1)
    
    dcosa=2.0/ Ncosa   
    Nki=len(Ki)  ;       
    Nkj=len(Kj)  ;  kj_min=min(Kj)  ;  kj_max=max(Kj)   
    WF_kikj=np.zeros((Nki,Nkj))   
    
    WF_rand[0][0]=0.0  ;  WF_rand[1][0]=1.0
    Pwin_spline = sp.interpolate.splrep(WF_rand[0],WF_rand[1], s=0) 
    
    Neps=epsilon_par[2];  eps_min=epsilon_par[0];  eps_max=epsilon_par[1]
    deps=(eps_max-eps_min)/Neps
    # https://wenku.baidu.com/view/719b47fa0740be1e640e9a97.html
    cosa=np.zeros((Ncosa+1))
    for i_cosa in range(Ncosa+1):
        cosa[i_cosa]=-1.0+i_cosa*dcosa
        
    eps=np.zeros((Neps+1))
    for i_eps in range(Neps+1):
        eps[i_eps]=eps_min+i_eps*deps
    Pwin = sp.interpolate.splev(eps, Pwin_spline, der=0)
    
    for i in range(Nki):
       FUN=np.zeros(  (Nkj,Neps+1,Ncosa+1)  ) 
       for i_eps in range(Neps+1):  
          for i_cosa in range(Ncosa+1):
             if (np.abs(Ki[i]**2+eps[i_eps]**2-2.0*Ki[i]*eps[i_eps]*cosa[i_cosa])<10.e-15) and (cosa[i_cosa]==1.0):
                 R_eps=0.0
             else:
                 R_eps= np.sqrt(Ki[i]**2+eps[i_eps]**2-2.0*Ki[i]*eps[i_eps]*cosa[i_cosa])
             bins = int( Nkj*(R_eps-kj_min)/(kj_max-kj_min) )
             if((bins >= 0) and (bins <Nkj)): 
                 FUN[bins,i_eps,i_cosa]= eps[i_eps]* eps[i_eps] * Pwin[i_eps]  
       #--------------------           
       for j in range(Nkj):    
          F1=FUN[j,0,0]+FUN[j,Neps,0]+FUN[j,0,Ncosa]+FUN[j,Neps,Ncosa] 
          F2=sum(FUN[j,:,0])    
          F3=sum(FUN[j,:,Ncosa])    
          F4=sum(FUN[j,0,:])    
          F5=sum(FUN[j,Neps,:]) 
          F6=sum(sum(FUN[j,1:Neps,1:Ncosa]))
          WF_kikj[i,j]=deps*dcosa*(0.25*F1+0.5*(F2+F3+F4+F5)+F6)           
    
    # normalization WF_kikj[i,j]:
    for i in range(len(WF_kikj[:,1])):
        WF_kikj[i,:]=WF_kikj[i,:]/sum(WF_kikj[i,:])
        
    return WF_kikj,Kj

def Pk_CONV_ross(Ki,WFkikj,Pkj,k_random,WF_random):       
    Psum=np.zeros(len(Ki))
    for i in range(len(Psum)):
        Psum[i]=sum(WFkikj[i,:]*Pkj)

    Pwin_spline = sp.interpolate.splrep(k_random,WF_random, s=0)
    PwinKi  = sp.interpolate.splev(Ki, Pwin_spline, der=0)
    Pwin0 = sp.interpolate.splev(0.0, Pwin_spline, der=0)

    P0 = sum(WFkikj[0,:]*Pkj) / Pwin0
    Pm = Psum - P0*PwinKi
    return Pm


# 5. Pypower convolution matrix:-----------------------------------------------
def ConvMat_Pypower( Xr,Yr,Zr, randoms_weights,Ngrid,wnorm,NrandRat,kmin,kmax,nkbin,kminc,kmaxc,nkbinc,ells ,boxsize,boxsizes,frac_nyq,pstype,WFCon_type=False):
  if(WFCon_type=='Pypower'):  
    from pypower import CatalogSmoothWindow,PowerSpectrumSmoothWindow,PowerSpectrumSmoothWindowMatrix,Projection 
    kout    = np.linspace(kmin,kmax,nkbin+1)[:-1]
    kout= kout[:-1]+0.5*np.diff(kout)
    edges = {'step': 2. * np.pi / boxsize}
    if(pstype!='crs'):
        wnorm=wnorm*NrandRat**2
        randoms_positions=np.zeros((len(Xr),3))
        randoms_positions[:,0]=Xr ; randoms_positions[:,1]=Yr ; randoms_positions[:,2]=Zr
        projss=[Projection(ell=0, wa_order=0), Projection(ell=2, wa_order=0), Projection(ell=4, wa_order=0), Projection(ell=6, wa_order=0),  Projection(ell=8, wa_order=0), 
                Projection(ell=1, wa_order=1), Projection(ell=3, wa_order=1), Projection(ell=5, wa_order=1), Projection(ell=7, wa_order=1)  ]
        window_large = CatalogSmoothWindow(randoms_positions1=randoms_positions, randoms_weights1=randoms_weights,
                                   projs=projss, nmesh=Ngrid,wnorm=wnorm, edges=edges, boxsize=boxsize,  boxcenter=np.array([0,0,0]),position_type='pos', dtype='f8').poles
        window_small = CatalogSmoothWindow(randoms_positions1=randoms_positions, randoms_weights1=randoms_weights,
                                   projs=projss, nmesh=Ngrid,wnorm=wnorm, edges=edges, boxsize=boxsizes, boxcenter=np.array([0,0,0]), position_type='pos', dtype='f8').poles
        window  = PowerSpectrumSmoothWindow.concatenate_x(window_large, window_small, frac_nyq=frac_nyq) # Here we remove the range above 0.9 Nyquist (which may not be reliable) 
        projsin = [Projection(ell=0, wa_order=0), Projection(ell=2, wa_order=0), Projection(ell=4, wa_order=0)]#,  
                   #Projection(ell=1, wa_order=1), Projection(ell=3, wa_order=1) ] 
    if(pstype=='crs'):
        Xr1,Xr2=Xr
        Yr1,Yr2=Yr
        Zr1,Zr2=Zr
        rand1=np.zeros((len(Xr1),3));rand2=np.zeros((len(Xr2),3))
        rand1[:,0]=Xr1 ; rand1[:,1]=Yr1 ; rand1[:,2]=Zr1
        rand2[:,0]=Xr2 ; rand2[:,1]=Yr2 ; rand2[:,2]=Zr2
        rweit1,rweit2=randoms_weights  
        Nr1,Nr2=NrandRat       
        wnorm=wnorm*Nr1*Nr2  
        projss=[Projection(ell=ell, wa_order=0) for ell in range(7)]
        window_large = CatalogSmoothWindow(randoms_positions1=rand1,randoms_positions2=rand2, randoms_weights1=rweit1,randoms_weights2=rweit2,
                                   projs=projss, nmesh=Ngrid,wnorm=wnorm, edges=edges, boxsize=boxsize,  boxcenter=np.array([0,0,0]), position_type='pos', dtype='f8').poles
        window_small = CatalogSmoothWindow(randoms_positions1=rand1,randoms_positions2=rand2, randoms_weights1=rweit1,randoms_weights2=rweit2,
                                   projs=projss, nmesh=Ngrid,wnorm=wnorm, edges=edges, boxsize=boxsizes, boxcenter=np.array([0,0,0]), position_type='pos', dtype='f8').poles
        window  = PowerSpectrumSmoothWindow.concatenate_x(window_large, window_small, frac_nyq=frac_nyq) # Here we remove the range above 0.9 Nyquist (which may not be reliable)    
        projsin = [Projection(ell=1, wa_order=0),Projection(ell=3, wa_order=0)]  
    sep     = np.geomspace(1e-4, 4e3, nkbinc *4 )
    wawm    = PowerSpectrumSmoothWindowMatrix(kout, projsin=projsin, projsout=ells , window=window,kin_lim=(kminc,kmaxc),   sep=sep )
    kin     = wawm.xin[0]
    if(pstype=='crs'):
        return  (wawm.value).T, kout,kin
    else:
        return  (wawm.value).T, kout,kin

# 6. add convolution matrix to model power spectrum:---------------------------
def Pk_CONV_multi(convmat,pkm,psyp):
    if(psyp!='crs'):
        pk0unconv,pk2unconv,pk4unconv=pkm
        pkunconvlst = np.concatenate((pk0unconv,pk2unconv,pk4unconv))
        pkconvlst   = np.dot(convmat,pkunconvlst)    
        nlmod=3
        N=len(pkconvlst)//nlmod 
        pk0conv,pk2conv,pk4conv = pkconvlst[:N],pkconvlst[N:2*N],pkconvlst[2*N:3*N]
        return pk0conv,pk2conv,pk4conv 
    if(psyp=='crs'):    
        pk1unconv,pk3unconv=pkm
        pkunconvlst = np.concatenate((pk1unconv,pk3unconv))
        pkconvlst   = np.dot(convmat,pkunconvlst)    
        nlmod=2
        N=len(pkconvlst)//nlmod 
        pk1conv,pk3conv = pkconvlst[:N],pkconvlst[N:2*N] 
        return pk1conv,pk3conv  
 
#########################     The end of Sec 3.    ############################ 
###############################################################################


















###############################################################################
######                                                                   ######
######       Sec 4. Convert power spectrum to correlation function       ######
######                                                                   ######
############################################################################### 
# Note: The following code is copied from the PYTHON package 'hankl', see:
# https://hankl.readthedocs.io/en/latest/install.html 
# or
# https://github.com/minaskar/hankl.git 
# you can install it using 
# pip install hankl
def Fun_preprocess(x, f, ext=0, range=None):    
    if range is not None:
        try:
            x_min, x_max = range
        except:
            raise TypeError(
                "Please enter valid x range in the form of a tuple (x_min, x_max) or list [x_min, x_max]."
            )
    else:
        x_min = None
        x_max = None

    try:
        ext_left, ext_right = ext
    except:
        ext_left = ext_right = ext

    x, f, N_left, N_right = Fun_padding(
        x, f, ext_left=ext_left, ext_right=ext_right, n_ext=0
    )

    if (x_min is not None) and (x_max is not None):

        if ext_left > 0 and ext_right > 0:
            while x[0] > x_min and x[-1] < x_max:
                x, f, N_left_prime, N_right_prime = Fun_padding(
                    x, f, ext_left=ext_left, ext_right=ext_right, n_ext=1
                )
                N_left += N_left_prime
                N_right += N_right_prime

    if x_min is not None:

        if ext_left > 0:
            while x[0] > x_min and (x_max is None or x[-1] >= x_max):
                x, f, N_left_prime, N_right_prime = Fun_padding(
                    x, f, ext_left=ext_left, ext_right=0, n_ext=1
                )
                N_left += N_left_prime
                N_right += N_right_prime

    if x_max is not None:

        if ext_right > 0:
            while x[-1] < x_max and (x_min is None or x[0] <= x_min):
                x, f, N_left_prime, N_right_prime = Fun_padding(
                    x, f, ext_left=0, ext_right=ext_right, n_ext=1
                )
                N_left += N_left_prime
                N_right += N_right_prime

    return x, f, N_left, N_right
def Fun_padding(x, f, ext_left=0, ext_right=0, n_ext=0):    
    N = x.size
    if N < 2:
        raise ValueError("Size of input arrays needs to be larger than 2")
    N_prime = 2 ** ((N - 1).bit_length() + n_ext)

    if N_prime > N:
        N_tails = N_prime - N

        if ext_left > 0 and ext_right > 0:
            N_left = N_tails // 2
            N_right = N_tails - N_left
        elif ext_left > 0 and ext_right < 1:
            N_left = N_tails
            N_right = 0
        elif ext_left < 1 and ext_right > 0:
            N_left = 0
            N_right = N_tails
        elif ext_left < 1 and ext_right < 1:
            return x, f, 0, 0
        else:
            raise ValueError(
                "Please provide valid values for ext argument (i.e. 0, 1, 2, 3)"
            )

        delta = (np.log10(np.max(x)) - np.log10(np.min(x))) / float(N - 1)
        x_prime = np.logspace(
            np.log10(x[0]) - N_left * delta, np.log10(x[-1]) + N_right * delta, N_prime
        )

        if N_left > 0:
            if ext_left == 1:
                f_left = np.zeros(N_left)
            elif ext_left == 2:
                f_left = np.full(N_left, f[0])
            elif ext_left == 3:
                f_left = f[0] * (f[1] / f[0]) ** np.arange(-N_left, 0)
            else:
                raise ValueError(
                    "Please provide valid values for ext argument (i.e. 0, 1, 2, 3)"
                )
        else:
            f_left = np.array([])

        if N_right > 0:
            if ext_right == 1:
                f_right = np.zeros(N_right)
            elif ext_right == 2:
                f_right = np.full(N_right, f[-1])
            elif ext_right == 3:
                f_right = f[-1] * (f[-1] / f[-2]) ** np.arange(1, N_right + 1)
            else:
                raise ValueError(
                    "Please provide valid values for ext argument (i.e. 0, 1, 2, 3)"
                )
        else:
            f_right = np.array([])

        f_prime = np.nan_to_num(np.concatenate((f_left, f, f_right)))

        return x_prime, f_prime, N_left, N_right

    else:
        return x, f, 0, 0
def Fun_gamma_term(mu, x, cutoff=200.0):    
    imag_x = np.imag(x)
    g_m = np.zeros(x.size, dtype=complex)
    asym_x = x[np.absolute(imag_x) > cutoff]
    asym_plus = (mu + 1 + asym_x) / 2.0
    asym_minus = (mu + 1 - asym_x) / 2.0
    x_good = x[(np.absolute(imag_x) <= cutoff) & (x != mu + 1.0 + 0.0j)]
    alpha_plus = (mu + 1.0 + x_good) / 2.0
    alpha_minus = (mu + 1.0 - x_good) / 2.0
    g_m[(np.absolute(imag_x) <= cutoff) & (x != mu + 1.0 + 0.0j)] = gamma(alpha_plus)/gamma(alpha_minus)
    # high-order expansion
    g_m[np.absolute(imag_x) > cutoff] = np.exp(
        (asym_plus - 0.5) * np.log(asym_plus)
        - (asym_minus - 0.5) * np.log(asym_minus)
        - asym_x
        + 1.0 / 12.0 * (1.0 / asym_plus - 1.0 / asym_minus)
        + 1.0 / 360.0 * (1.0 / asym_minus ** 3.0 - 1.0 / asym_plus ** 3.0)
        + 1.0 / 1260.0 * (1.0 / asym_plus ** 5.0 - 1.0 / asym_minus ** 5.0)   )
    g_m[np.where(x == mu + 1.0 + 0.0j)[0]] = 0.0 + 0.0j
    return g_m
def Fun_lowring_xy(mu, q, L, N, xy=1.0):   
    delta_L = L / float(N)
    x = q + 1j * np.pi / delta_L
    x_plus = (mu + 1 + x) / 2.0
    x_minus = (mu + 1 - x) / 2.0
    phip = np.imag(np.log(gamma(x_plus)))
    phim = np.imag(np.log(gamma(x_minus)))
    arg = np.log(2.0 / xy) / delta_L + (phip - phim) / np.pi
    iarg = np.rint(arg)
    if arg != iarg:
        xy = xy * np.exp((arg - iarg) * delta_L)
    return xy
def Fun_u_m_term(m, mu, q, xy, L, cutoff=200.0):    
    omega = 1j * 2 * np.pi * m / float(L)
    x = q + omega
    U_mu = 2 ** x * Fun_gamma_term(mu, x, cutoff)
    u_m = (xy) ** (-omega) * U_mu
    u_m[m.size - 1] = np.real(u_m[m.size - 1])
    return u_m
def FFTLog_Fun(x, f_x, q, mu, xy=1.0, lowring=False, ext=0, range=None, return_ext=False, stirling_cutoff=200.0):   
    if mu + 1.0 + q == 0.0:
        raise ValueError("The FFTLog Hankel Transform is singular when mu + 1 + q = 0.")
    x, f_x, N_left, N_right = Fun_preprocess(x, f_x, ext=ext, range=range)
    N = f_x.size
    delta_L = (np.log(np.max(x)) - np.log(np.min(x))) / float(N - 1)
    L = np.log(np.max(x)) - np.log(np.min(x))
    log_x0 = np.log(x[N // 2])
    x0 = np.exp(log_x0)
    c_m = np.fft.rfft(f_x)
    m = np.fft.rfftfreq(N, d=1.0) * float(N)
    if lowring:
        xy = Fun_lowring_xy(mu, q, L, N, xy)
    y0 = xy / x0
    log_y0 = np.log(y0)
    m_y = np.arange(-N // 2, N // 2)
    m_shift = np.fft.fftshift(m_y)
    s = delta_L * (-m_y) + log_y0
    id = m_shift
    y = 10 ** (s[id] / np.log(10))
    u_m = Fun_u_m_term(m, mu, q, xy, L, stirling_cutoff)
    b = c_m * u_m
    A_m = np.fft.irfft(b)
    f_y = A_m[id]
    f_y = f_y[::-1]
    y = y[::-1]
    if q != 0:
        f_y = f_y * (y) ** (-float(q))
    if return_ext:
        return y, f_y
    else:
        if N_right == 0:
            return y[N_left:], f_y[N_left:]
        else:
            return y[N_left:-N_right], f_y[N_left:-N_right]        
def Ps_to_Xi_Fun(k, P, l, n=0, lowring=False, ext=0, range=None, return_ext=False, stirling_cutoff=200.0):
    r, f = FFTLog_Fun( k, P * k ** 1.5, q=-n, mu=l + 0.5, lowring=lowring, ext=ext, range=range, return_ext=return_ext, stirling_cutoff=stirling_cutoff )
    return r, f * (2.0 * np.pi) ** (-1.5) * r ** (-1.5) * (1j) ** l
def Xi_to_Ps_Fun(r, xi, l, n=0, lowring=False, ext=0, range=None, return_ext=False, stirling_cutoff=200.0):
    k, F = FFTLog_Fun( r, xi * r ** 1.5, q=n, mu=l + 0.5, lowring=lowring, ext=ext, range=range, return_ext=return_ext, stirling_cutoff=stirling_cutoff )
    return k, F * (2.0 * np.pi) ** 1.5 * k ** (-1.5) * (-1j) ** l        
def get_extrap_pl_Fun(logk_ar, kog, data, pfun):
    kmax = kog[-1]
    k2ext = logk_ar[np.where(logk_ar>kmax)]
    k2spline = logk_ar[np.where(logk_ar<=kmax)]
    a, b = get_pl_coefs_Fun(kog[-2:], data[-2:])
    pkextended = a*(k2ext**b)
    pksplined = pfun(k2spline)
    return np.hstack([pksplined,pkextended])
def get_pl_coefs_Fun(x, y):
    b = np.log(y[0]/y[1])/np.log(x[0]/x[1])
    a = y[1]/(x[1]**b)
    return a, b
def PKmod2xi_Fun(klog, k_ext, kmulti, pkmulti, ell, kmax, kinv = False):
    #kmulti, pkmulti = PS multipole k and P(k)^ell values
    #ell             = desired multipole
    #klog            = array of equally spaced k values in logspace over the approximate k range of Fei's results 
    #kmax            = maximum k value to use in power law extrapolation
    #k_ext           = full k range to extrapolate to
    #k-inv - when calculating the ell = 1 component of the momemtum CF model, we need to correct for a (kr) term hardcoded in hankl
    #      - and so instead of (k, P(k)) the arguments are (k, P(k)/k). 1/r component comes after call to hankl.P2xi
    idkmax = np.where(klog < kmax)
    fun = interpolate.interp1d(kmulti, pkmulti)
    fun_log = fun(klog)
    fun_ext = get_extrap_pl_Fun(k_ext, klog[idkmax], fun_log[idkmax], fun)
    if kinv:
        rmulti, ximulti = Ps_to_Xi_Fun(k_ext, fun_ext/k_ext, l = ell, lowring = True, ext = 3)
    else:
        rmulti, ximulti = Ps_to_Xi_Fun(k_ext, fun_ext, l = ell, lowring = True, ext = 3)
    return rmulti, ximulti
def CFmodMulti_Fun(r, k, Pk , l,   kcut, PS_type):
    # see https://arxiv.org/pdf/2207.03707  convert PS to CF
    klog  = np.logspace(np.log10(np.min(k)), np.log10(np.max(k)), 2**11)
    k_ext = np.logspace(np.log10(np.min(k)), 3, 2**11) 
    if(PS_type =='den'):
        r0, Xigg  = PKmod2xi_Fun(klog, k_ext, k, Pk , ell = l, kmax = kcut, kinv = False) # final clustering monopole    
        return r , np.interp(r, r0,  Xigg )
    if(PS_type =='mom'):
        r0, pp0 = PKmod2xi_Fun(klog, k_ext, k, 3.*Pk, ell = 0, kmax = kcut, kinv = False)
        pp0     = np.interp(r, r0,  pp0 )
        r0, pp1 = PKmod2xi_Fun(klog, k_ext, k, 3.*Pk, ell = 1, kmax = kcut, kinv = True)
        pp1     = np.interp(r, r0,  pp1 )
        psiperp = np.imag(pp1)/r #1/r correction for hankl
        psipar  = pp0 - 2.*np.imag(pp1)/r
        return r, psiperp, psipar
    if(PS_type =='crs'):    
        r0, gp0 = PKmod2xi_Fun(klog, k_ext, k, Pk, ell = 1, kmax = kcut, kinv = False)
        gp0     = np.interp(r, r0,  gp0 )
        gv_dip  = np.imag(gp0)
        return  r,  gv_dip
    
#########################     The end of Sec 4.    ############################# 
################################################################################


