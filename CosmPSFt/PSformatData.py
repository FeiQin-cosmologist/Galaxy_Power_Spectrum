import numpy as np
import scipy as sp
from scipy import integrate
from scipy.interpolate import splev, splrep


###############################################################################
######                                                                   ######
######   Sec 1. The Functions used to calculate cosmoligical distance    ######
######                                                                   ######
###############################################################################
LightSpeed = 299792.458
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
def spline_Rsf2Dis(OmegaM,OmegaA,Hub,nbin=10000,redmax=2.5):
    dist = np.empty(nbin);red = np.empty(nbin)
    for j in range(nbin):
        red[j] = j*redmax/nbin
        dist[j] = DistDc(red[j],OmegaM,OmegaA, 0.0,Hub,-1.0, 0.0, 0.0)
    dist_spline = sp.interpolate.splrep(red, dist, s=0)
    return dist_spline
def DisRsfConvert(xdt,OmegaM,OmegaA,Hub,nbin=10000,redmax=2.5):
    spl_fun=spline_Rsf2Dis(OmegaM,OmegaA,Hub,nbin,redmax)
    Distc        = splev(xdt, spl_fun)
    return Distc
def Sky2Cat(ra,dec,rsft,OmegaM , OmegaA ,Hub,nbin=3000,redmax=2.5):
    disz= DisRsfConvert(rsft,OmegaM,OmegaA,Hub,nbin,redmax);rsft=[]
    X = disz*np.cos(dec/180.*np.pi)*np.cos(ra/180.*np.pi)
    Y = disz*np.cos(dec/180.*np.pi)*np.sin(ra/180.*np.pi)
    Z = disz*np.sin(dec/180.*np.pi)    
    return X,Y,Z 
#  peculiar velocity estimators:-----------------------------------------------
# we use Watkins&Feldman2015 :   https://arxiv.org/abs/1411.6665 
def Vp_to_logd(vpec,Rsf,OmegaM):
    deccel = 3.0*OmegaM/2.0 - 1.0
    Vmod   = Rsf*LightSpeed*(1.0 + 0.5*(1.0 - deccel)*Rsf - (2.0 - deccel - 3.0*deccel*deccel)*Rsf*Rsf/6.0)
    Logd   =    vpec / ( np.log(10.)*Vmod/(1.0+Vmod/LightSpeed)  )
    return Logd
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
# galaxy number density:-------------------------------------------------------
def nbarS_Fun(delt_z,zhR,weit,survey_area,OmegaM,OmegaA,Hub,nbin=10000,redmax=2.3):
    Nz=int(np.abs(np.max(zhR)-np.min(zhR))/delt_z)
    Binz=np.zeros((Nz+2))
    for i in range(Nz+2):
        Binz[i]=np.min(zhR)+i*delt_z
    if type(weit) is not np.ndarray:
        weit  = np.ones(len(zhR))*weit*1.
    HIS=np.histogram(zhR,bins=Binz,weights =weit)
    N=HIS[0]
    B=HIS[1]
    Dz=DisRsfConvert(B,OmegaM,OmegaA,Hub,nbin,redmax)
    spl_nbar  = np.array([0.]*(len(Dz)-1))
    for i in range(len(Dz)-1):
        Rin  = Dz[i]
        Rout = Dz[i+1]
        volume = (survey_area)*(Rout*Rout*Rout-Rin*Rin*Rin)/3.0 # survey_area in unit of pi not degree
        spl_nbar[i] = N[i]/volume;
    if(len(spl_nbar)==1):
      nbs=np.ones(len(zhR))*spl_nbar
    else:
      if(len(spl_nbar)<=3)and (len(spl_nbar)>=2):
        nbar_spline = sp.interpolate.splrep(Dz[0:len(Dz)-1] + np.abs(Dz[1]-Dz[2])/2.,spl_nbar, k=len(spl_nbar)-1)
      else:
        nbar_spline = sp.interpolate.splrep(Dz[0:len(Dz)-1] + np.abs(Dz[1]-Dz[2])/2.,spl_nbar, s=0)
      Dzs=DisRsfConvert(zhR,OmegaM,OmegaA,Hub,nbin=10000,redmax=0.3)
      nbs=sp.interpolate.splev(Dzs, nbar_spline, der=0)
    return nbs
def nbarSAsign_Fun(rsf,zmR,nbR_Snorm):
    ind = np.argsort(zmR);
    x   = zmR[ind]
    y   = nbR_Snorm[ind]
    nb  = np.interp(rsf, x,y)
    return nb
def nbarG_Fun(ra,dec,rsf,weit,nx,ny,nz,OmegaM,OmegaA,Hub):
    rsfmax    = np.max(rsf)
    distmax   = DisRsfConvert(rsfmax,OmegaM,OmegaA,Hub) 
    lx,ly,lz  = 2.*distmax,2.*distmax,2.*distmax
    x0,y0,z0  = distmax,distmax,distmax
    dx,dy,dz  = lx/nx,ly/ny,lz/nz
    dvol      = dx*dy*dz
    xlims     = np.linspace(0.,lx,nx+1) - x0
    ylims     = np.linspace(0.,ly,ny+1) - y0
    zlims     = np.linspace(0.,lz,nz+1) - z0
    # Convert to (x,y,z) positions
    ndat      = len(ra)*1.
    x,y,z     = Sky2Cat(ra,dec,rsf ,OmegaM , OmegaA ,Hub)
    ra,dec,rsf= [],[],[]
    # Create number density catalogue
    if type(weit) is not np.ndarray:
        Num,edges = np.histogramdd(np.vstack([x+x0,y+y0,z+z0]).transpose(),bins=(nx,ny,nz),range=((0.,lx),(0.,ly),(0.,lz)) )
    else:
        Num,edges = np.histogramdd(np.vstack([x+x0,y+y0,z+z0]).transpose(),bins=(nx,ny,nz),range=((0.,lx),(0.,ly),(0.,lz)),weights =weit)
        weit =[]
    ndensgrid = (ndat/dvol)*(Num/np.sum(Num))
    # Sample number density at random positions
    ix        = np.digitize(x,xlims) - 1 ; x=[]
    iy        = np.digitize(y,ylims) - 1 ; y=[]
    iz        = np.digitize(z,zlims) - 1 ; z=[]
    nb        = ndensgrid[ix,iy,iz]
    return nb,ndensgrid,xlims,ylims,zlims 
def nbarGAsign_Fun(ra,dec,rsf,nbRgrid_norm,xlims,ylims,zlims,OmegaM,OmegaA,Hub):
    x,y,z     = Sky2Cat(ra,dec,rsf ,OmegaM , OmegaA ,Hub)
    ix        = np.digitize(x,xlims) - 1
    iy        = np.digitize(y,ylims) - 1
    iz        = np.digitize(z,zlims) - 1
    nb        = nbRgrid_norm[ix,iy,iz]
    return nb 
#########################     The end of Sec 1.    ############################ 
############################################################################### 


    

def FormatData_Fun(Datas,Output_dir,Datype):
    if(Datype=='gal-survey')or(Datype=='gal-mock'):
        Ra,Dec,cz,nb,FKPden=Datas    
        w=1.*np.ones(len(Ra))
        wfkpden =1./(   1.+nb*FKPden)
        ndata=len(np.where(nb>0)[0])
        outfile = open(Output_dir, 'w')
        outfile.write("#   %18d\n"% (ndata))
        for i in range(len(Ra)):
          if(nb[i]>0):  
            outfile.write(" %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf \n" % (Ra[i],Dec[i],cz[i],nb[i],w[i]*wfkpden[i] ))  
        outfile.close()
        wfkp= wfkpden 
    if(Datype=='gal-rand'):
        Ra,Dec,cz,nb,FKPden =Datas
        w=1.*np.ones(len(Ra))
        wfkpden  = 1./(   1.+nb*FKPden)
        ndata=len(np.where(nb>0)[0]) 
        outfile = open(Output_dir, 'w')
        outfile.write("#   %18d\n"% (ndata))
        for i in range(len(Ra)):
          if(nb[i]>0):  
            outfile.write(" %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf \n" % (Ra[i],Dec[i],cz[i], nb[i],w[i]*wfkpden[i] ))  
        outfile.close()    
        wfkp= wfkpden 
    if(Datype=='pv-survey')or(Datype=='pv-mock'):
        Ra,Dec,cz,logd,elogd,nb,sig_v ,FKPden, FKPmom=Datas    
        w=1.*np.ones(len(Ra))
        wfkpmom =1./(   (cz/(1.+cz/299792.458)*np.log(10.)*elogd)**2+sig_v**2 + nb*FKPmom   )
        wfkpden =1./(   1.+nb*FKPden)
        ndata=len(np.where(nb>0)[0])
        outfile = open(Output_dir, 'w')
        outfile.write("#   %18d\n"% (ndata))
        for i in range(len(Ra)):
          if(nb[i]>0):  
            outfile.write(" %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %21.18lf  %12.6lf \n" % (Ra[i],Dec[i],cz[i],logd[i],nb[i],w[i]*wfkpmom[i],w[i]*wfkpden[i] ))  
        outfile.close()
        wfkp=[wfkpden,wfkpmom]
    if(Datype=='pv-rand'):
        Ra,Dec,cz,pvr,epv,nb,sig_v,FKPden, FKPmom=Datas
        w=1.*np.ones(len(Ra))
        wfkpmom  = 1./(epv**2+sig_v**2+ nb*FKPmom) 
        wfkpden =1./(   1.+nb*FKPden)
        ndata=len(np.where(nb>0)[0]) 
        outfile = open(Output_dir, 'w')
        outfile.write("#   %18d\n"% (ndata))
        for i in range(len(Ra)):
          if(nb[i]>0):  
            outfile.write(" %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %21.18lf  %12.6lf \n" % (Ra[i],Dec[i],cz[i], pvr[i], nb[i],w[i]*wfkpmom[i],w[i]*wfkpden[i] ))  
        outfile.close() 
        wfkp=[wfkpden,wfkpmom]
    return Output_dir,wfkp

