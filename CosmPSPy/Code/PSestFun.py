import numpy as np
import scipy as sp
from scipy import integrate
from scipy.interpolate import splev, splrep
import scipy.fft
import sys

  

   


def np_save(file_dir,X,pic=4):
    import pickle      
    pickle.dump(X, open(file_dir, 'wb'), protocol=pic)
    return [] 
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

























###############################################################################
#####                                                                     #####
#####   Sec 2. The Functions used to calculate measured power spectrum    #####
#####                                                                     #####
###############################################################################
def GridCorr_Fun(dx,dy,dz,fx,fy,fz):
    sincx = np.ones(len(fx)) ;  sincy = np.ones(len(fy)) ;  sincz = np.ones(len(fz))
    indx  = (fx != 0.)       ;  indy  = (fy != 0.)       ;  indz  = (fz != 0.)
    sincx[indx] = np.sin( fx[indx]*dx*np.pi )/( fx[indx]*dx*np.pi );
    sincy[indy] = np.sin( fy[indy]*dy*np.pi )/( fy[indy]*dy*np.pi );
    sincz[indz] = np.sin( fz[indz]*dz*np.pi )/( fz[indz]*dz*np.pi );    
    grid_cor    = 1.  / ( sincx*sincy*sincz ) 
    return grid_cor                     

def Pkest_Fun(kmin,kmax,nk,nx,ny,nz,xmin,xmax,ymin,ymax,zmin,zmax,  
              longi =0,lati =0,rsf =0,nb=0,FKP=1600.,  
              longir=0,latir=0,rsfr=0,nbr_norm=0, 
              longiv=0,lativ=0,rsfv=0,vp=0,evp=0,nbmom=0,FKPv=5.*10.**9,   
              OmegaM=0.3,OmegaA=0.7,Hub=100.,sigv=300.0,
              bulk_vel=np.array([0.,0.,0.]),  
              PS_type='mom', PS_multi='yes', file_dir='PSestDir' ,wd=1.,wdR=1.,wv=1.):
    #PS_multi= 'yes' , 'no' and 'all' for nont-zero multiples, monople and all multiples enven it is zero. 
    #PS_type = 'den' , 'mom' and 'crs' for density, momentum and cross power spectrum, respectivly. 
    #==========================================================================
    #1: Initiall settings: 
    # 1.1 check nx ny nz are even number: 
    if((nx % 2)!=0): print('\n Error: nx should set to be a even number.\n') ;  sys.exit()
    if((ny % 2)!=0): print('\n Error: ny should set to be a even number.\n') ;  sys.exit()
    if((nz % 2)!=0): print('\n Error: nz should set to be a even number.\n') ;  sys.exit()
    print(' ', 'Please make sure that in the catalogue, observer is at Coordinate origin=[0,0,0] \n'  )
    # 1.2 number of galaxies in the catalogue:
    if((PS_type == 'den') or (PS_type == 'crs')): ndata   = len(longi) ;  print(' ', PS_type,'  Ngal=',ndata  ) 
    if((PS_type == 'mom') or (PS_type == 'crs')): ndatapv = len(vp)    ;  print(' ', PS_type,'  Npv =',ndatapv)   
    print(' ' )
    # 1.3 grid size:
    lx = np.abs(xmax-xmin) ; ly = np.abs(ymax-ymin) ; lz = np.abs(zmax-zmin) 
    # 1.4 set k=bin:
    dk = (kmax-kmin)/nk    ; Nbink = np.zeros(nk,dtype=int) 
    # 1.5 define arrays to store the PS multiples: 
    p0 = np.zeros(nk)
    if((PS_multi!='no')):
        p1 = np.zeros(nk) ; p2 = np.zeros(nk) ; p3 = np.zeros(nk) ; p4 = np.zeros(nk)        
    # 1.6 convert to Cartesian coordinate and FKP weights:
    if((PS_type == 'den') or (PS_type == 'crs')):
        nbr     = nbr_norm
        x,y,z   = Sky2Cat(longi ,lati ,rsf ,OmegaM,OmegaA,Hub)
        xr,yr,zr= Sky2Cat(longir,latir,rsfr,OmegaM,OmegaA,Hub)
        w       = 1./(1.+nb *FKP)
        wr      = 1./(1.+nbr*FKP)
    if((PS_type == 'mom') or (PS_type == 'crs')):
        xpv,ypv,zpv= Sky2Cat(longiv,lativ,rsfv,OmegaM,OmegaA,Hub)
        wmom    = 1./(evp**2+sigv**2 + nbmom*FKPv   )
        wrho    = 1./(1.+nbmom*FKP)
        if(FKPv==0):wmom=1.*np.ones(len(xpv))
    # 1.8 remove bulk flow velociy:
    if((PS_type == 'mom') or (PS_type == 'crs')):
        if((bulk_vel[0]!=0.)or(bulk_vel[1]!=0.)or(bulk_vel[2]!=0.)):
            print('  ', 'Bulk velocity removed.\n')
            Dis  = np.sqrt(  xpv**2+ypv**2+zpv**2 )
            hatx = xpv/Dis
            haty = ypv/Dis
            hatz = zpv/Dis
            vp   = vp-(bulk_vel[0]*hatx +bulk_vel[1]*haty +bulk_vel[2]*hatz)

    #==========================================================================
    #2. Make grid of the galaxies- FUN:  
    dx =(xmax-xmin)/nx ;  dy = (ymax-ymin)/ny ;  dz = (zmax-zmin)/nz
    #2.1: assign galaxies to grids:
    if((PS_type == 'den') or (PS_type == 'crs')):
        FUN,edges = np.histogramdd(np.vstack([x,y,z]).transpose()   , bins=(nx,ny,nz), range=((-lx/2.,lx/2.),(-ly/2.,ly/2.),(-lz/2.,lz/2.)), weights=w        * wd      )
        alpha     = np.sum(w)/np.sum(wr)
        tmp,edges = np.histogramdd(np.vstack([xr,yr,zr]).transpose(), bins=(nx,ny,nz), range=((-lx/2.,lx/2.),(-ly/2.,ly/2.),(-lz/2.,lz/2.)), weights=alpha*wr * wdR     )
        FUN       = FUN-tmp 
    if((PS_type == 'mom') or (PS_type == 'crs')):
        if(PS_type == 'mom'):
            FUN,edges    = np.histogramdd(np.vstack([xpv,ypv,zpv]).transpose(), bins=(nx,ny,nz), range=((-lx/2.,lx/2.),(-ly/2.,ly/2.),(-lz/2.,lz/2.)), weights=wmom*vp  *wv )
        if(PS_type == 'crs'):
            FUNmom,edges = np.histogramdd(np.vstack([xpv,ypv,zpv]).transpose(), bins=(nx,ny,nz), range=((-lx/2.,lx/2.),(-ly/2.,ly/2.),(-lz/2.,lz/2.)), weights=wmom*vp  *wv )
    edges=0.;tmp=0.
    #2.2: nyquist frequency f_nqu of the grid: 
    fx_nqu = np.pi/dx ; fy_nqu = np.pi/dy  ;  fz_nqu = np.pi/dz ;  min_nqu = fx_nqu 
    if(fy_nqu<min_nqu): min_nqu = fy_nqu   ;
    if(fz_nqu<min_nqu): min_nqu = fz_nqu   ;
    #2.3: possitions of grid cells:
    if(PS_multi!='no'): 
        gx        = np.linspace(xmin,xmax,nx+1)[:-1] ;  gy   = np.linspace(ymin,ymax,ny+1)[:-1] ;  gz = np.linspace(zmin,zmax,nz+1)[:-1];    
        gd        = np.meshgrid(gx,gy,gz)            ;  vect = np.zeros((nx*ny*nz,3))
        # vect = [ vect_Z, vect_x, vect_y ],  Z-component is the first element.
        vect[:,1] = gd[0].flatten() ;
        vect[:,2] = gd[1].flatten() ;
        vect[:,0] = gd[2].flatten() ;  
        Lvect     = vect[:,0]*vect[:,0]+vect[:,1]*vect[:,1]+vect[:,2]*vect[:,2]  
        Lvect[Lvect==0.0] = 1.0 ;  gd=0; gx=0; gy=0; gz=0               
    # in k-space: 
    kvect      = np.zeros((nx*ny*nz//2,3)) ;  grid_cor = np.zeros( nx*ny*nz//2 )   
    kv0        = np.concatenate((np.linspace(0.,2.*np.pi*(nx//2)/(nx*dx),  nx//2+1),np.linspace(2.*np.pi*(nx//2+1-nx)/(nx*dx),0., nx//2)[:-1]))
    kv1        = np.concatenate((np.linspace(0.,2.*np.pi*(ny//2)/(ny*dy),  ny//2+1),np.linspace(2.*np.pi*(ny//2+1-ny)/(ny*dy),0., ny//2)[:-1]))
    kv2        = np.linspace(0.,2.*np.pi*(nz//2-1)/(nz*dz),nz//2)
    gdk        = np.meshgrid(kv0,kv1,kv2) ; 
    # kvect = [ kvect_Z, kvect_x, kvect_y ],  Z-component is the first element.
    kvect[:,1] = gdk[0].flatten()
    kvect[:,2] = gdk[1].flatten()
    kvect[:,0] = gdk[2].flatten() 
    grid_cor   = GridCorr_Fun(dx,dy,dz,kvect[:,1]/(2.*np.pi),kvect[:,2]/(2.*np.pi),kvect[:,0]/(2.*np.pi)) 
    kw         = np.sqrt(kvect[:,0]*kvect[:,0]+kvect[:,1]*kvect[:,1]+kvect[:,2]*kvect[:,2]); 
    # nyquist frequency
    ikw        = np.array((kw-kmin)/dk,dtype=int)
    ind_nqu    = np.where((ikw>=0)&(ikw<nk)&(kw<(0.5*min_nqu)))[0]
    induse     = []
    for ikbin in range(nk):   
        ind    = np.where(ikw[ind_nqu]==ikbin)[0] 
        induse.append( ind_nqu[ind])    
 
    #===========================================================================
    #3. Normalization factor and shot noise: 
    if(PS_type =='den'):
        Pnoise = np.sum(w*w *wd*wd)             ; PnoiseR = alpha*alpha*np.sum(wr*wr* wdR*wdR) ;  Norm = np.sum(nb*w*w *wd*wd)  
        PSN    = (Pnoise+PnoiseR)/Norm
    if (PS_type=='mom'):
        Pvnoise= np.sum(wmom*wmom*vp*vp *wv*wv) ; Norm = np.sum(nbmom*wmom*wmom *wv*wv)  
        PSN    = Pvnoise/Norm
    if (PS_type=='crs'):
        PnoiseC= np.sum(wrho*wmom*vp *wv   ) ; Norm = np.sqrt( np.sum(nb*w*w * wd*wd))*np.sqrt(np.sum(nbmom*wmom*wmom *wv*wv)) 
        PSN    = 0.
    
    #==========================================================================        
    #4. l=0 : 
    # 4.1: Fourier transformation:
    FUN_out = scipy.fft.fftn( FUN )[:,:,:nz//2].flatten() # np.fft.fftn( FUN )[:,:,:nz//2].flatten()
    if (PS_type=='crs'): FUN_outmom = scipy.fft.fftn( FUNmom )[:,:,:nz//2].flatten() # np.fft.fftn( FUNmom )[:,:,:nz//2].flatten() 
    if ((PS_type=='crs')and(PS_multi=='all')): tmp1 = (0.5*(np.real(FUN_out)*np.imag(FUN_outmom)-np.imag(FUN_out)*np.real(FUN_outmom)+np.real(FUN_outmom)*np.imag(FUN_out)-np.imag(FUN_outmom)*np.real(FUN_out))-PnoiseC)*grid_cor*grid_cor 
    if (PS_type =='den'):                      tmp2 = (     np.real(FUN_out)**2+np.imag(FUN_out)**2-(Pnoise+PnoiseR))*grid_cor*grid_cor
    if (PS_type =='mom'):                      tmp3 = (     np.real(FUN_out)**2+np.imag(FUN_out)**2        - Pvnoise)*grid_cor*grid_cor
    if((PS_multi=='yes')or(PS_multi=='all')):
        if(PS_type    != 'crs'):
            tmp4       = (np.real(FUN_out)**2+np.imag(FUN_out)**2)*grid_cor*grid_cor
        if((PS_type   == 'crs')and(PS_multi=='all')):
            tmp5       = 0.5*( np.real(FUN_out) * np.imag(FUN_outmom) -np.imag(FUN_out) *np.real(FUN_outmom) +np.real(FUN_outmom) *np.imag(FUN_out) -np.imag(FUN_outmom) *np.real(FUN_out)  ) *grid_cor*grid_cor
    # 4.2: l=0,2,4: 
    for ikbin in range(nk):
        INDuse=induse[ikbin];
        if ((PS_type=='crs')and(PS_multi=='all')): p0[ikbin] = np.sum(tmp1[INDuse])
        if (PS_type =='den'):                      p0[ikbin] = np.sum(tmp2[INDuse])
        if (PS_type =='mom'):                      p0[ikbin] = np.sum(tmp3[INDuse])
        Nbink[ikbin]= len(ikw[INDuse])
        # calculate the l=2,4 power spectrum: 
        if((PS_multi=='yes')or(PS_multi=='all')):
            if(PS_type != 'crs'):
                tmp       = np.sum(tmp4[INDuse])
                p2[ikbin] = -1./2.*tmp  ;  p4[ikbin] = 0.375*tmp
            if((PS_type   == 'crs')and(PS_multi=='all')):
                tmp       = np.sum(tmp5[INDuse])
                p2[ikbin] = -1./2.*tmp  ;  p4[ikbin] = 0.375* tmp                
    # 4.3 save the original density grid without FFT:              
    FUN_save    = FUN + 0. -0.       ; FUN     = np.zeros((nx,ny,nz))
    # save the FFT of the density grid :
    FUN_out_save= FUN_out  + 0. - 0. ; FUN_out = 0.
    tmp1=0.;tmp2=0.;tmp3=0.;tmp4=0.;tmp5=0.;
    # save the grid of velcity field: 
    if(PS_type == 'crs'):
        FUN_savemom     = FUNmom + 0. - 0.     ; FUNmom     = np.zeros((nx,ny,nz))
        FUN_out_savemom = FUN_outmom + 0. - 0. ; FUN_outmom = 0.   
       
    #============       If you want to calculate the PS multiples ============= 
    if(PS_multi!='no'):
    #========================================================================== 
    #5. l=1:  
        # 5.1 project the grid to mod functions: 
        if(PS_type=='crs')or(PS_multi=='all'): 
            for ii in range(3): 
                if(PS_type!='crs'): FUN = FUN_save    * (vect[:,ii]/np.sqrt(Lvect)).reshape(nx,ny,nz)  
                if(PS_type=='crs'): 
                    FUN    =              FUN_save    * (vect[:,ii]/np.sqrt(Lvect)).reshape(nx,ny,nz)     
                    FUNmom =              FUN_savemom * (vect[:,ii]/np.sqrt(Lvect)).reshape(nx,ny,nz)                 
        # 5.2: calculate FFT
                if( (PS_type != 'crs')and(PS_multi=='all')):  
                    FUN_out   = scipy.fft.fftn( FUN )[:,:,:nz//2].flatten()
                if(PS_type    =='crs'):
                    FUN_out   = scipy.fft.fftn( FUN )[:,:,:nz//2].flatten()  ;  FUN_outmom = scipy.fft.fftn( FUNmom )[:,:,:nz//2].flatten()  
                if( (PS_type != 'crs')and(PS_multi=='all')):
                    tmp1      = kvect[:,ii]*((np.real(FUN_out_save)*np.imag(FUN_out)-np.imag(FUN_out_save)*np.real(FUN_out))*grid_cor*grid_cor)                                        
                if(PS_type    == 'crs'):   
                    tmp2      = kvect[:,ii]*(0.5*(-np.imag(FUN_out)*np.real(FUN_out_savemom)+np.real(FUN_out)*np.imag(FUN_out_savemom)+np.imag(FUN_outmom)*np.real(FUN_out_save)-np.real(FUN_outmom)*np.imag(FUN_out_save))* grid_cor*grid_cor) 
        # 5.3 calculate l=1,3:
                if(PS_type=='crs')or(PS_multi=='all'):     
                    for ikbin in range(nk):
                        # calculate the l=1,3 power spectrum:         
                        kprefac = 1.0; 
                        INDuse=induse[ikbin]; KW=kw[INDuse]; indkw=(KW>0); 
                        if( (PS_type != 'crs')and(PS_multi=='all')):
                            p1[ikbin] = p1[ikbin] +np.sum(( 1.0*kprefac* tmp1[INDuse] )[indkw]/KW[indkw] )                    
                            p3[ikbin] = p3[ikbin] +np.sum((-1.5*kprefac* tmp1[INDuse] )[indkw]/KW[indkw] )                
                        if(PS_type    == 'crs'):   
                            p1[ikbin] = p1[ikbin] +np.sum(( 1.0*kprefac* tmp2[INDuse] )[indkw]/KW[indkw] )
                            p3[ikbin] = p3[ikbin] +np.sum((-1.5*kprefac* tmp2[INDuse] )[indkw]/KW[indkw] )                                                                                                                                                             
        # 5.4 save data and clear RMA:  
            tmp1=0.;tmp2=0.
            if((PS_type != 'crs')and(PS_multi=='all')): FUN=np.zeros((nx,ny,nz));FUN_out=0.
            if( PS_type == 'crs'): FUNmom= np.zeros((nx,ny,nz));FUN=np.zeros((nx,ny,nz));FUN_out=0.;FUN_outmom=0.
 
    #==========================================================================    
    # 6. l=2: 
        # 6.1, grid data project to mod functions:
        if(PS_type!='crs')or(PS_multi=='all'):     
            for ii in range(3):
                for jj in range(3):
                   if(jj>=ii):
                       if(PS_type!='crs'): FUN = FUN_save    * (vect[:,ii]*vect[:,jj]/Lvect ).reshape(nx,ny,nz)  
                       if(PS_type=='crs'):
                           FUN    =              FUN_save    * (vect[:,ii]*vect[:,jj]/Lvect ).reshape(nx,ny,nz)    
                           FUNmom =              FUN_savemom * (vect[:,ii]*vect[:,jj]/Lvect ).reshape(nx,ny,nz)  
        # 6.2: calculate FFT:
                       FUN_out = scipy.fft.fftn( FUN )[:,:,:nz//2].flatten()
                       if((PS_type=='crs')and(PS_multi=='all')):  FUN_outmom = scipy.fft.fftn( FUNmom )[:,:,:nz//2].flatten()  
                       if (PS_type   != 'crs'):
                           tmp1   = kvect[:,ii]*kvect[:,jj]*(     (np.real(FUN_out)*np.real(FUN_out_save)+np.imag(FUN_out)*np.imag(FUN_out_save))*grid_cor*grid_cor)                                                                 
                       if((PS_type   == 'crs')and(PS_multi=='all')): 
                           tmp2   = kvect[:,ii]*kvect[:,jj]*( 0.5*(np.real(FUN_out)*np.imag(FUN_out_savemom)-np.imag(FUN_out)*np.real(FUN_out_savemom)+np.real(FUN_outmom)*np.imag(FUN_out_save)-np.imag(FUN_outmom)*np.real(FUN_out_save))*grid_cor*grid_cor)                                                
        # 6.3 calculate l=2: 
                       for ikbin in range(nk):   
                           kprefac = 1.0;
                           if (ii != jj):kprefac = 2.0;
                           INDuse=induse[ikbin]; KW=kw[INDuse]; indkw=(KW>0);  
                           if (PS_type  != 'crs'):
                               p2[ikbin] = p2[ikbin]+np.sum(( 1.5 *kprefac * tmp1[INDuse] )[indkw]/KW[indkw]**2 )                                                              
                               p4[ikbin] = p4[ikbin]+np.sum((-3.75*kprefac * tmp1[INDuse] )[indkw]/KW[indkw]**2 )                                                                  
                           if((PS_type   == 'crs')and(PS_multi=='all')): 
                               p2[ikbin] = p2[ikbin]+np.sum(( 1.5 *kprefac * tmp2[INDuse] )[indkw]/KW[indkw]**2 )
                               p4[ikbin] = p4[ikbin]+np.sum((-3.75*kprefac * tmp2[INDuse] )[indkw]/KW[indkw]**2 )                                                 
            FUN=np.zeros((nx,ny,nz));FUN_out=0.
            if(PS_type == 'crs'): FUNmom=np.zeros((nx,ny,nz));FUN_out=0.;FUN_outmom=0.
            tmp1=0.;tmp2=0.
     
    #==========================================================================
    # 7. l=3:
        # 7.1, grid data project to mod functions:
        if(PS_type=='crs')or(PS_multi=='all'):     
            jusm=0
            for ii in range(3):
                for jj in range(3):
                    for kk in range(3):
                        if((jusm != 3) and (jusm != 4) and (jusm != 6) and (jusm != 7) and (jusm != 8))and((jusm != 9) and (jusm != 10) and (jusm != 11) and (jusm != 15) and (jusm != 16))and((jusm != 17) and (jusm != 18) and (jusm != 19) and (jusm != 20) and (jusm != 21))and((jusm != 22) and (jusm != 23)):
                            if(PS_type!='crs'):FUN = FUN_save    * (vect[:,ii]*vect[:,jj]*vect[:,kk]/(Lvect*np.sqrt(Lvect) ) ).reshape(nx,ny,nz)                  
                            if(PS_type=='crs'):
                                FUN                = FUN_save    * (vect[:,ii]*vect[:,jj]*vect[:,kk]/(Lvect*np.sqrt(Lvect) ) ).reshape(nx,ny,nz)                 
                                FUNmom             = FUN_savemom * (vect[:,ii]*vect[:,jj]*vect[:,kk]/(Lvect*np.sqrt(Lvect) ) ).reshape(nx,ny,nz)
        #7.2: calculate FFT:
                            if( (PS_type !='crs')and(PS_multi=='all')):  
                                FUN_out   = scipy.fft.fftn( FUN )[:,:,:nz//2].flatten()
                            if(PS_type   =='crs'):
                                FUN_out   = scipy.fft.fftn( FUN )[:,:,:nz//2].flatten() ; FUN_outmom = scipy.fft.fftn( FUNmom )[:,:,:nz//2].flatten()  
                            if( (PS_type != 'crs')and(PS_multi=='all')):
                                tmp1      = kvect[:,ii]* kvect[:,jj]* kvect[:,kk]*((np.real(FUN_out_save)  * np.imag(FUN_out)-np.imag(FUN_out_save) * np.real(FUN_out))* grid_cor*grid_cor)                                                                 
                            if (PS_type   =='crs'):       
                                tmp2      = kvect[:,ii]* kvect[:,jj]* kvect[:,kk]*(0.5*(-np.imag(FUN_out) * np.real(FUN_out_savemom)+np.real(FUN_out)*np.imag(FUN_out_savemom)+np.imag(FUN_outmom)*np.real(FUN_out_save)-np.real(FUN_outmom)*np.imag(FUN_out_save))*grid_cor*grid_cor)                                
        #7.3 calculate l=3:                                         
                            for ikbin in range(nk):  
                                # calculate the l=3 power spectrum:       
                                kprefac = 1.0;  
                                if (ii != kk) :
                                    if (ii != jj): kprefac = 6.0;
                                    else:          kprefac = 3.0;
                                INDuse=induse[ikbin]; KW=kw[INDuse]; indkw=(KW>0);  
                                if( (PS_type != 'crs')and(PS_multi=='all')):
                                    p3[ikbin] = p3[ikbin]+np.sum((2.5*kprefac* tmp1[INDuse])[indkw]/KW[indkw]**3)                                                                
                                if (PS_type   =='crs'):       
                                    p3[ikbin] = p3[ikbin]+np.sum((2.5*kprefac* tmp2[INDuse])[indkw]/KW[indkw]**3)                               
                        jusm=jusm+1
            if( (PS_type != 'crs')and(PS_multi=='all')): FUN=np.zeros((nx,ny,nz));FUN_out=0.
            if(PS_type =='crs'): FUNmom= np.zeros((nx,ny,nz)); FUN=np.zeros((nx,ny,nz));FUN_out=0.;FUN_outmom=0.
            tmp1=0.;tmp2=0.
        
    #==========================================================================
    #8. l=4 
        if(PS_type!='crs')or(PS_multi=='all'):
            jusm=1
            for ii in range(3):
               for jj in range(3):
                  for kk in range(3):
                      if((jusm != 4)  and  (jusm != 7)  and (jusm != 8)  and (jusm != 10)) and ((jusm != 11) and  (jusm != 16) and (jusm != 17) and (jusm != 19)) and ((jusm != 21) and  (jusm != 22) and (jusm != 23) and (jusm != 24)) :
                          if(PS_type!='crs'):  FUN = FUN_save    * (vect[:,ii]*vect[:,ii]*vect[:,jj]*vect[:,kk]/Lvect**2 ).reshape(nx,ny,nz)
                          if(PS_type=='crs'):
                              FUN                  = FUN_save    * (vect[:,ii]*vect[:,ii]*vect[:,jj]*vect[:,kk]/Lvect**2 ).reshape(nx,ny,nz)
                              FUNmom               = FUN_savemom * (vect[:,ii]*vect[:,ii]*vect[:,jj]*vect[:,kk]/Lvect**2 ).reshape(nx,ny,nz)                                   
        #8.2: calculate FFT
                          FUN_out = scipy.fft.fftn( FUN )[:,:,:nz//2].flatten()
                          if((PS_type=='crs')and(PS_multi=='all')): FUN_outmom = scipy.fft.fftn( FUNmom )[:,:,:nz//2].flatten()  
                          if (PS_type  != 'crs'):
                              tmp1      = kvect[:,ii]*kvect[:,ii]*kvect[:,jj]*kvect[:,kk]*((np.real(FUN_out) * np.real(FUN_out_save) + np.imag(FUN_out) * np.imag(FUN_out_save) ) * grid_cor*grid_cor) 
                          if ((PS_type == 'crs')and(PS_multi=='all')):
                              tmp2      = kvect[:,ii]*kvect[:,ii]*kvect[:,jj]*kvect[:,kk]*(0.5*( np.real(FUN_out) * np.imag(FUN_out_savemom) -np.imag(FUN_out) * np.real(FUN_out_savemom) +np.real(FUN_outmom)  * np.imag(FUN_out_save) - np.imag(FUN_outmom) * np.real(FUN_out_save) )* grid_cor*grid_cor)   
        #8.3 calculate l=4:                                         
                          for ikbin in range(nk):                           
                              kprefac = 1.0;
                              if (ii == jj):
                                  if (ii != kk): kprefac = 4.0;
                              else: 
                                  if (jj == kk): kprefac = 6.0;
                                  else :kprefac = 12.0;
                              INDuse=induse[ikbin]; KW=kw[INDuse] ; indkw=(KW>0);    
                              if (PS_type  != 'crs'):
                                  p4[ikbin] = p4[ikbin]+np.sum((4.375*kprefac * tmp1[INDuse])[indkw]/KW[indkw]**4 )
                              if ((PS_type == 'crs')and(PS_multi=='all')):
                                  p4[ikbin] = p4[ikbin]+np.sum((4.375*kprefac * tmp2[INDuse])[indkw]/KW[indkw]**4 )        
                      jusm=jusm+1           
            tmp1=0.;tmp2=0.
    #====================        The end of PS multiples ======================
    #9. save data -------
    # normalize ps: 
    for ikw in range(nk):
        if( Nbink[ikw]>0. ):
            p0[ikw] = p0[ikw]/(Nbink[ikw]*Norm);
            if((PS_multi=='yes')or(PS_multi=='all')):
               p1[ikw] = p1[ikw]*3./(Nbink[ikw]*Norm);
               p2[ikw] = p2[ikw]*5./(Nbink[ikw]*Norm);
               p3[ikw] = p3[ikw]*7./(Nbink[ikw]*Norm);
               p4[ikw] = p4[ikw]*9./(Nbink[ikw]*Norm);    
    #set the last k-bin to be 0:  
    Nbink[-1]=0.
    # save data:
    outfile = open(file_dir, 'w')
    outfile.write("# No.          k            P0            P1           P2           P3           P4            Nk           Norm           SNois  \n") 
    if((PS_multi!='no')):
      for i in range(nk):
        if(i==nk-1):
            outfile.write("  %7d     %17.10lf     %7d     %7d     %7d     %7d     %7d     %7d     %7d     %7d \n"%(i+1, kmin+(i+0.5)*dk,0,0,0,0,0,0,0,0))
        else:
            outfile.write("  %7d     %17.10lf     %17.10lf     %17.10lf     %17.10lf     %17.10lf     %17.10lf    %7d     %30.20lf     %30.20lf \n"%(i+1, kmin+(i+0.5)*dk,np.real(p0[i]),np.real(p1[i]),np.real(p2[i]),np.real(p3[i]),np.real(p4[i]),Nbink[i],Norm,PSN))
    if(PS_multi=='no'):
      for i in range(nk):
        if(i==nk-1):
          outfile.write("  %7d     %17.10lf     %7d     %7d     %7d     %7d     %7d     %7d     %7d     %7d \n"%(i+1, kmin+(i+0.5)*dk,0,0,0,0,0,0,0,0))
        else:
          outfile.write("  %7d     %17.10lf     %17.10lf     %7d     %7d     %7d     %7d     %7d     %30.20lf     %30.20lf \n"%(i+1, kmin+(i+0.5)*dk,np.real(p0[i]),0,0,0,0,Nbink[i],Norm,PSN))
    outfile.close()   
    if(PS_type == 'den'):
        weifkp=[w,wr]
    if(PS_type == 'mom'):
        weifkp=[wmom,wrho]
    if(PS_type == 'crs'): 
        weifkp=[w,wr,wmom,wrho]
    return  file_dir,weifkp,Norm   

def PkestR_Fun(kmin,kmax,nk,nx,ny,nz,xmin,xmax,ymin,ymax,zmin,zmax,longir,latir,rsfr,epv,nbr_norm,FKP,OmegaM,OmegaA,Hub,sigv,PS_type,file_dir='PSestDir', wdt=1.):
    if((nx % 2)!=0):
        print('\n Error: nx should set to be a even number.\n')
        sys.exit()
    if((ny % 2)!=0):
        print('\n Error: ny should set to be a even number.\n')
        sys.exit()
    if((nz % 2)!=0):
        print('\n Error: nz should set to be a even number.\n')
        sys.exit()     
    print('\n  Random-PS=', PS_type,'\n')   
    xr,yr,zr=Sky2Cat(longir,latir,rsfr,OmegaM,OmegaA,Hub)
    # ndata is the number of galaxies rather than randoms:
    nbr     = nbr_norm
    if(PS_type=='den'): wr=1./(1.+nbr*FKP)
    if(PS_type=='mom'): wr=1./(epv**2+sigv**2+nbr*FKP) 
    if(FKP==0):wr=1.*np.ones(len(xr))
    # 1: grid data:
    dx=(xmax-xmin)/nx; dy=(ymax-ymin)/ny; dz=(zmax-zmin)/nz
    lx=np.abs(xmax-xmin) ; ly=np.abs(ymax-ymin) ; lz=np.abs(zmax-zmin) 
    FUN,edges = np.histogramdd(np.vstack([xr,yr,zr]).transpose(),bins=(nx,ny,nz),range=((-lx/2.,lx/2.),(-ly/2.,ly/2.),(-lz/2.,lz/2.)),weights= wr * wdt)
    # 2: shotnoise:
    PnoiseR= np.sum(wr*wr * wdt)
    # 3: nyquist frequency f_nqu of the grid: 
    fx_nqu=np.pi/dx  ;  fy_nqu=np.pi/dy  ;  fz_nqu=np.pi/dz  ;  min_nqu=fx_nqu;
    if(fy_nqu<min_nqu): min_nqu=fy_nqu;
    if(fz_nqu<min_nqu): min_nqu=fz_nqu;
    # 4: calculate FFT
    FUN_out = scipy.fft.fftn(FUN )  # np.fft.fftn( FUN )
    # 5. calculate l=0 :
    dk=(kmax-kmin)/nk ; Nbink=np.zeros(nk,dtype=int) ; p0=np.zeros(nk)
    for i in range(nx):
        if(i<=nx//2): fx = i      / (nx*dx)
        else:         fx = (i-nx) / (nx*dx)
        for j in range(ny):
            if(j<=ny//2):fy = j      / (ny*dy)
            else:        fy = (j-ny) / (ny*dy)
            for k in range(nz//2):
                fz  = k/(nz*dz)
                kw  = 2.*np.pi*np.sqrt(fx*fx+fy*fy+fz*fz)
                ikw = int((kw-kmin)/dk)
                if( (ikw>=0)and(ikw<nk)and(kw<(0.5*min_nqu)) ):
                    sincx=1.;sincy=1.;sincz=1.
                    if(fx != 0.): sincx = np.sin(fx*dx*np.pi)/(fx*dx*np.pi);
                    if(fy != 0.): sincy = np.sin(fy*dy*np.pi)/(fy*dy*np.pi);
                    if(fz != 0.): sincz = np.sin(fz*dz*np.pi)/(fz*dz*np.pi);
                    grid_cor=1./(sincx*sincy*sincz);
                    p0[ikw] = p0[ikw]+(np.real(FUN_out[i,j,k])**2+np.imag(FUN_out[i,j,k])**2-PnoiseR)*grid_cor*grid_cor
                    if(ikw==0): NormR=np.real(FUN_out[i,j,k])**2 +np.imag(FUN_out[i,j,k])**2-PnoiseR
                    Nbink[ikw]=Nbink[ikw]+1
    # 6. normalize: 
    for ikw in range(nk):
        if( Nbink[ikw]>0. ):
            p0[ikw] = p0[ikw]/(Nbink[ikw]*NormR);        
    Nbink[-1]=0.
    outfile = open(file_dir, 'w')
    for i in range(nk):
        if(np.real(p0[i])>0.):
            if(i==nk-1):
              outfile.write("  %7d     %17.10lf     %7d     %7d \n"%(i+1,kmin+(i+0.5)*dk,0,0))
            else:
              outfile.write("  %7d     %17.10lf     %17.10lf     %7d \n"%(i+1,kmin+(i+0.5)*dk,np.real(p0[i]),Nbink[i]))
    outfile.close()
    return  file_dir,wr    

 
def Pkest_SimBox_Fun(kmin,kmax,nk,nx,ny,nz,Lbox,  
              x,y,z,vx,vy,vz,   
              OmegaM=0.3,OmegaA=0.7,Hub=100. , 
              PS_type='mom', PS_multi='yes', file_dir='PSestDir' ,wd=1., wv=1.):
    #1: Initiall settings: 
    # 1.1 check nx ny nz are even number: 
    if((nx % 2)!=0): print('\n Error: nx should set to be a even number.\n') ;  sys.exit()
    if((ny % 2)!=0): print('\n Error: ny should set to be a even number.\n') ;  sys.exit()
    if((nz % 2)!=0): print('\n Error: nz should set to be a even number.\n') ;  sys.exit()
    print(' ', 'Please make sure that the Coordinate origin=[0,0,0] is in the center of the sim-box.\n'  )
    # 1.2 number of galaxies in the catalogue:
    ndata   = len(x) ;  print(' ', PS_type,'  Ngal=',ndata  ) 
    # 1.3 grid size:
    lx =   ly =   lz = Lbox
    dx =Lbox/nx ;  dy = Lbox/ny ;  dz = Lbox/nz
    xmin,xmax = -lx/2.,lx/2.  
    ymin,ymax = -ly/2.,ly/2. 
    zmin,zmax = -lz/2.,lz/2. 
    # 1.4 set k=bin:
    dk = (kmax-kmin)/nk    ; Nbink = np.zeros(nk,dtype=int) 
    # 1.5 define arrays to store the PS multiples: 
    p0 = np.zeros(nk)
    if((PS_multi!='no')):
        p1 = np.zeros(nk) ; p2 = np.zeros(nk) ; p3 = np.zeros(nk) ; p4 = np.zeros(nk)        
     

    #==========================================================================
    #2. Make grid of the galaxies- FUN:  
    dx =Lbox/nx ;  dy = Lbox/ny ;  dz = Lbox/nz
    #2.1: assign galaxies to grids:
    if((PS_type == 'den') or (PS_type == 'crs')):
        FUN,edges = np.histogramdd(np.vstack([x,y,z]).transpose()   , bins=(nx,ny,nz), range=((-lx/2.,lx/2.),(-ly/2.,ly/2.),(-lz/2.,lz/2.)), weights=  wd*np.ones(len(x))         )
        FUN       = FUN-ndata/ (nx*ny*nz) 
    if((PS_type == 'mom') or (PS_type == 'crs')):
        vp = (vx*x +vy*y +vz*z ) / np.sqrt(x*x+y*y+z*z) 
        if(PS_type == 'mom'):
            FUN,edges    = np.histogramdd(np.vstack([x,y,z]).transpose(), bins=(nx,ny,nz), range=((-lx/2.,lx/2.),(-ly/2.,ly/2.),(-lz/2.,lz/2.)), weights= vp  *wv )
        if(PS_type == 'crs'):
            FUNmom,edges = np.histogramdd(np.vstack([x,y,z]).transpose(), bins=(nx,ny,nz), range=((-lx/2.,lx/2.),(-ly/2.,ly/2.),(-lz/2.,lz/2.)), weights= vp  *wv )
    edges=0. 
    #2.2: nyquist frequency f_nqu of the grid: 
    fx_nqu = np.pi/dx ; fy_nqu = np.pi/dy  ;  fz_nqu = np.pi/dz ;  min_nqu = fx_nqu 
    if(fy_nqu<min_nqu): min_nqu = fy_nqu   ;
    if(fz_nqu<min_nqu): min_nqu = fz_nqu   ;
    #2.3: possitions of grid cells:
    if(PS_multi!='no'): 
        gx        = np.linspace(xmin,xmax,nx+1)[:-1] ;  gy   = np.linspace(ymin,ymax,ny+1)[:-1] ;  gz = np.linspace(zmin,zmax,nz+1)[:-1];    
        gd        = np.meshgrid(gx,gy,gz)            ;  vect = np.zeros((nx*ny*nz,3))
        # vect = [ vect_Z, vect_x, vect_y ],  Z-component is the first element.
        vect[:,1] = gd[0].flatten() ;
        vect[:,2] = gd[1].flatten() ;
        vect[:,0] = gd[2].flatten() ;  
        Lvect     = vect[:,0]*vect[:,0]+vect[:,1]*vect[:,1]+vect[:,2]*vect[:,2]  
        Lvect[Lvect==0.0] = 1.0 ;  gd=0; gx=0; gy=0; gz=0               
    # in k-space: 
    kvect      = np.zeros((nx*ny*nz//2,3)) ;  grid_cor = np.zeros( nx*ny*nz//2 )   
    kv0        = np.concatenate((np.linspace(0.,2.*np.pi*(nx//2)/(nx*dx),  nx//2+1),np.linspace(2.*np.pi*(nx//2+1-nx)/(nx*dx),0., nx//2)[:-1]))
    kv1        = np.concatenate((np.linspace(0.,2.*np.pi*(ny//2)/(ny*dy),  ny//2+1),np.linspace(2.*np.pi*(ny//2+1-ny)/(ny*dy),0., ny//2)[:-1]))
    kv2        = np.linspace(0.,2.*np.pi*(nz//2-1)/(nz*dz),nz//2)
    gdk        = np.meshgrid(kv0,kv1,kv2) ; 
    # kvect = [ kvect_Z, kvect_x, kvect_y ],  Z-component is the first element.
    kvect[:,1] = gdk[0].flatten()
    kvect[:,2] = gdk[1].flatten()
    kvect[:,0] = gdk[2].flatten() 
    grid_cor   = GridCorr_Fun(dx,dy,dz,kvect[:,1]/(2.*np.pi),kvect[:,2]/(2.*np.pi),kvect[:,0]/(2.*np.pi)) 
    kw         = np.sqrt(kvect[:,0]*kvect[:,0]+kvect[:,1]*kvect[:,1]+kvect[:,2]*kvect[:,2]); 
    # nyquist frequency
    ikw        = np.array((kw-kmin)/dk,dtype=int)
    ind_nqu    = np.where((ikw>=0)&(ikw<nk)&(kw<(0.5*min_nqu)))[0]
    induse     = []
    for ikbin in range(nk):   
        ind    = np.where(ikw[ind_nqu]==ikbin)[0] 
        induse.append( ind_nqu[ind])    
 
    #==========================================================================
    #3. Normalization factor and shot noise: 
    Norm = ndata**2/Lbox**3     
    if(PS_type =='den'):
        Pnoise = ndata +0.-0.   ; PSN    = Pnoise/Norm
    if (PS_type=='mom'):
        Pvnoise= np.sum( vp*vp *wv*wv)  ; PSN    = Pvnoise/Norm
    if (PS_type=='crs'):
        PnoiseC= np.sum( vp *wv   ) ;   PSN    = 0.
        
    #==========================================================================        
    #4. l=0 : 
    # 4.1: Fourier transformation:
    FUN_out = scipy.fft.fftn( FUN )[:,:,:nz//2].flatten() # np.fft.fftn( FUN )[:,:,:nz//2].flatten()
    if (PS_type=='crs'): FUN_outmom = scipy.fft.fftn( FUNmom )[:,:,:nz//2].flatten() # np.fft.fftn( FUNmom )[:,:,:nz//2].flatten() 
    if ((PS_type=='crs')and(PS_multi=='all')): tmp1 = (0.5*(np.real(FUN_out)*np.imag(FUN_outmom)-np.imag(FUN_out)*np.real(FUN_outmom)+np.real(FUN_outmom)*np.imag(FUN_out)-np.imag(FUN_outmom)*np.real(FUN_out))-PnoiseC)*grid_cor*grid_cor 
    if (PS_type =='den'):                      tmp2 = (     np.real(FUN_out)**2+np.imag(FUN_out)**2- Pnoise )*grid_cor*grid_cor
    if (PS_type =='mom'):                      tmp3 = (     np.real(FUN_out)**2+np.imag(FUN_out)**2        - Pvnoise)*grid_cor*grid_cor
    if((PS_multi=='yes')or(PS_multi=='all')):
        if(PS_type    != 'crs'):
            tmp4       = (np.real(FUN_out)**2+np.imag(FUN_out)**2)*grid_cor*grid_cor
        if((PS_type   == 'crs')and(PS_multi=='all')):
            tmp5       = 0.5*( np.real(FUN_out) * np.imag(FUN_outmom) -np.imag(FUN_out) *np.real(FUN_outmom) +np.real(FUN_outmom) *np.imag(FUN_out) -np.imag(FUN_outmom) *np.real(FUN_out)  ) *grid_cor*grid_cor
    # 4.2: l=0,2,4: 
    for ikbin in range(nk):
        INDuse=induse[ikbin];
        if ((PS_type=='crs')and(PS_multi=='all')): p0[ikbin] = np.sum(tmp1[INDuse])
        if (PS_type =='den'):                      p0[ikbin] = np.sum(tmp2[INDuse])
        if (PS_type =='mom'):                      p0[ikbin] = np.sum(tmp3[INDuse])
        Nbink[ikbin]= len(ikw[INDuse])
        # calculate the l=2,4 power spectrum: 
        if((PS_multi=='yes')or(PS_multi=='all')):
            if(PS_type != 'crs'):
                tmp       = np.sum(tmp4[INDuse])
                p2[ikbin] = -1./2.*tmp  ;  p4[ikbin] = 0.375*tmp
            if((PS_type   == 'crs')and(PS_multi=='all')):
                tmp       = np.sum(tmp5[INDuse])
                p2[ikbin] = -1./2.*tmp  ;  p4[ikbin] = 0.375* tmp                
    # 4.3 save the original density grid without FFT:              
    FUN_save    = FUN + 0. -0.       ; FUN     = np.zeros((nx,ny,nz))
    # save the FFT of the density grid :
    FUN_out_save= FUN_out  + 0. - 0. ; FUN_out = 0.
    tmp1=0.;tmp2=0.;tmp3=0.;tmp4=0.;tmp5=0.;
    # save the grid of velcity field: 
    if(PS_type == 'crs'):
        FUN_savemom     = FUNmom + 0. - 0.     ; FUNmom     = np.zeros((nx,ny,nz))
        FUN_out_savemom = FUN_outmom + 0. - 0. ; FUN_outmom = 0.   
       
    #============       If you want to calculate the PS multiples ============= 
    if(PS_multi!='no'):
    #========================================================================== 
    #5. l=1:  
        # 5.1 project the grid to mod functions: 
        if(PS_type=='crs')or(PS_multi=='all'): 
            for ii in range(3): 
                if(PS_type!='crs'): FUN = FUN_save    * (vect[:,ii]/np.sqrt(Lvect)).reshape(nx,ny,nz)  
                if(PS_type=='crs'): 
                    FUN    =              FUN_save    * (vect[:,ii]/np.sqrt(Lvect)).reshape(nx,ny,nz)     
                    FUNmom =              FUN_savemom * (vect[:,ii]/np.sqrt(Lvect)).reshape(nx,ny,nz)                 
        # 5.2: calculate FFT
                if( (PS_type != 'crs')and(PS_multi=='all')):  
                    FUN_out   = scipy.fft.fftn( FUN )[:,:,:nz//2].flatten()
                if(PS_type    =='crs'):
                    FUN_out   = scipy.fft.fftn( FUN )[:,:,:nz//2].flatten()  ;  FUN_outmom = scipy.fft.fftn( FUNmom )[:,:,:nz//2].flatten()  
                if( (PS_type != 'crs')and(PS_multi=='all')):
                    tmp1      = kvect[:,ii]*((np.real(FUN_out_save)*np.imag(FUN_out)-np.imag(FUN_out_save)*np.real(FUN_out))*grid_cor*grid_cor)                                        
                if(PS_type    == 'crs'):   
                    tmp2      = kvect[:,ii]*(0.5*(-np.imag(FUN_out)*np.real(FUN_out_savemom)+np.real(FUN_out)*np.imag(FUN_out_savemom)+np.imag(FUN_outmom)*np.real(FUN_out_save)-np.real(FUN_outmom)*np.imag(FUN_out_save))* grid_cor*grid_cor) 
        # 5.3 calculate l=1,3:
                if(PS_type=='crs')or(PS_multi=='all'):     
                    for ikbin in range(nk):
                        # calculate the l=1,3 power spectrum:         
                        kprefac = 1.0; 
                        INDuse=induse[ikbin]; KW=kw[INDuse]; indkw=(KW>0); 
                        if( (PS_type != 'crs')and(PS_multi=='all')):
                            p1[ikbin] = p1[ikbin] +np.sum(( 1.0*kprefac* tmp1[INDuse] )[indkw]/KW[indkw] )                    
                            p3[ikbin] = p3[ikbin] +np.sum((-1.5*kprefac* tmp1[INDuse] )[indkw]/KW[indkw] )                
                        if(PS_type    == 'crs'):   
                            p1[ikbin] = p1[ikbin] +np.sum(( 1.0*kprefac* tmp2[INDuse] )[indkw]/KW[indkw] )
                            p3[ikbin] = p3[ikbin] +np.sum((-1.5*kprefac* tmp2[INDuse] )[indkw]/KW[indkw] )                                                                                                                                                             
        # 5.4 save data and clear RMA:  
            tmp1=0.;tmp2=0.
            if((PS_type != 'crs')and(PS_multi=='all')): FUN=np.zeros((nx,ny,nz));FUN_out=0.
            if( PS_type == 'crs'): FUNmom= np.zeros((nx,ny,nz));FUN=np.zeros((nx,ny,nz));FUN_out=0.;FUN_outmom=0.
 
    #==========================================================================    
    # 6. l=2: 
        # 6.1, grid data project to mod functions:
        if(PS_type!='crs')or(PS_multi=='all'):     
            for ii in range(3):
                for jj in range(3):
                   if(jj>=ii):
                       if(PS_type!='crs'): FUN = FUN_save    * (vect[:,ii]*vect[:,jj]/Lvect ).reshape(nx,ny,nz)  
                       if(PS_type=='crs'):
                           FUN    =              FUN_save    * (vect[:,ii]*vect[:,jj]/Lvect ).reshape(nx,ny,nz)    
                           FUNmom =              FUN_savemom * (vect[:,ii]*vect[:,jj]/Lvect ).reshape(nx,ny,nz)  
        # 6.2: calculate FFT:
                       FUN_out = scipy.fft.fftn( FUN )[:,:,:nz//2].flatten()
                       if((PS_type=='crs')and(PS_multi=='all')):  FUN_outmom = scipy.fft.fftn( FUNmom )[:,:,:nz//2].flatten()  
                       if (PS_type   != 'crs'):
                           tmp1   = kvect[:,ii]*kvect[:,jj]*(     (np.real(FUN_out)*np.real(FUN_out_save)+np.imag(FUN_out)*np.imag(FUN_out_save))*grid_cor*grid_cor)                                                                 
                       if((PS_type   == 'crs')and(PS_multi=='all')): 
                           tmp2   = kvect[:,ii]*kvect[:,jj]*( 0.5*(np.real(FUN_out)*np.imag(FUN_out_savemom)-np.imag(FUN_out)*np.real(FUN_out_savemom)+np.real(FUN_outmom)*np.imag(FUN_out_save)-np.imag(FUN_outmom)*np.real(FUN_out_save))*grid_cor*grid_cor)                                                
        # 6.3 calculate l=2: 
                       for ikbin in range(nk):   
                           kprefac = 1.0;
                           if (ii != jj):kprefac = 2.0;
                           INDuse=induse[ikbin]; KW=kw[INDuse]; indkw=(KW>0);  
                           if (PS_type  != 'crs'):
                               p2[ikbin] = p2[ikbin]+np.sum(( 1.5 *kprefac * tmp1[INDuse] )[indkw]/KW[indkw]**2 )                                                              
                               p4[ikbin] = p4[ikbin]+np.sum((-3.75*kprefac * tmp1[INDuse] )[indkw]/KW[indkw]**2 )                                                                  
                           if((PS_type   == 'crs')and(PS_multi=='all')): 
                               p2[ikbin] = p2[ikbin]+np.sum(( 1.5 *kprefac * tmp2[INDuse] )[indkw]/KW[indkw]**2 )
                               p4[ikbin] = p4[ikbin]+np.sum((-3.75*kprefac * tmp2[INDuse] )[indkw]/KW[indkw]**2 )                                                 
            FUN=np.zeros((nx,ny,nz));FUN_out=0.
            if(PS_type == 'crs'): FUNmom=np.zeros((nx,ny,nz));FUN_out=0.;FUN_outmom=0.
            tmp1=0.;tmp2=0.
     
    #==========================================================================
    # 7. l=3:
        # 7.1, grid data project to mod functions:
        if(PS_type=='crs')or(PS_multi=='all'):     
            jusm=0
            for ii in range(3):
                for jj in range(3):
                    for kk in range(3):
                        if((jusm != 3) and (jusm != 4) and (jusm != 6) and (jusm != 7) and (jusm != 8))and((jusm != 9) and (jusm != 10) and (jusm != 11) and (jusm != 15) and (jusm != 16))and((jusm != 17) and (jusm != 18) and (jusm != 19) and (jusm != 20) and (jusm != 21))and((jusm != 22) and (jusm != 23)):
                            if(PS_type!='crs'):FUN = FUN_save    * (vect[:,ii]*vect[:,jj]*vect[:,kk]/(Lvect*np.sqrt(Lvect) ) ).reshape(nx,ny,nz)                  
                            if(PS_type=='crs'):
                                FUN                = FUN_save    * (vect[:,ii]*vect[:,jj]*vect[:,kk]/(Lvect*np.sqrt(Lvect) ) ).reshape(nx,ny,nz)                 
                                FUNmom             = FUN_savemom * (vect[:,ii]*vect[:,jj]*vect[:,kk]/(Lvect*np.sqrt(Lvect) ) ).reshape(nx,ny,nz)
        #7.2: calculate FFT:
                            if( (PS_type !='crs')and(PS_multi=='all')):  
                                FUN_out   = scipy.fft.fftn( FUN )[:,:,:nz//2].flatten()
                            if(PS_type   =='crs'):
                                FUN_out   = scipy.fft.fftn( FUN )[:,:,:nz//2].flatten() ; FUN_outmom = scipy.fft.fftn( FUNmom )[:,:,:nz//2].flatten()  
                            if( (PS_type != 'crs')and(PS_multi=='all')):
                                tmp1      = kvect[:,ii]* kvect[:,jj]* kvect[:,kk]*((np.real(FUN_out_save)  * np.imag(FUN_out)-np.imag(FUN_out_save) * np.real(FUN_out))* grid_cor*grid_cor)                                                                 
                            if (PS_type   =='crs'):       
                                tmp2      = kvect[:,ii]* kvect[:,jj]* kvect[:,kk]*(0.5*(-np.imag(FUN_out) * np.real(FUN_out_savemom)+np.real(FUN_out)*np.imag(FUN_out_savemom)+np.imag(FUN_outmom)*np.real(FUN_out_save)-np.real(FUN_outmom)*np.imag(FUN_out_save))*grid_cor*grid_cor)                                
        #7.3 calculate l=3:                                         
                            for ikbin in range(nk):  
                                # calculate the l=3 power spectrum:       
                                kprefac = 1.0;  
                                if (ii != kk) :
                                    if (ii != jj): kprefac = 6.0;
                                    else:          kprefac = 3.0;
                                INDuse=induse[ikbin]; KW=kw[INDuse]; indkw=(KW>0);  
                                if( (PS_type != 'crs')and(PS_multi=='all')):
                                    p3[ikbin] = p3[ikbin]+np.sum((2.5*kprefac* tmp1[INDuse])[indkw]/KW[indkw]**3)                                                                
                                if (PS_type   =='crs'):       
                                    p3[ikbin] = p3[ikbin]+np.sum((2.5*kprefac* tmp2[INDuse])[indkw]/KW[indkw]**3)                               
                        jusm=jusm+1
            if( (PS_type != 'crs')and(PS_multi=='all')): FUN=np.zeros((nx,ny,nz));FUN_out=0.
            if(PS_type =='crs'): FUNmom= np.zeros((nx,ny,nz)); FUN=np.zeros((nx,ny,nz));FUN_out=0.;FUN_outmom=0.
            tmp1=0.;tmp2=0.
        
    #==========================================================================
    #8. l=4 
        if(PS_type!='crs')or(PS_multi=='all'):
            jusm=1
            for ii in range(3):
               for jj in range(3):
                  for kk in range(3):
                      if((jusm != 4)  and  (jusm != 7)  and (jusm != 8)  and (jusm != 10)) and ((jusm != 11) and  (jusm != 16) and (jusm != 17) and (jusm != 19)) and ((jusm != 21) and  (jusm != 22) and (jusm != 23) and (jusm != 24)) :
                          if(PS_type!='crs'):  FUN = FUN_save    * (vect[:,ii]*vect[:,ii]*vect[:,jj]*vect[:,kk]/Lvect**2 ).reshape(nx,ny,nz)
                          if(PS_type=='crs'):
                              FUN                  = FUN_save    * (vect[:,ii]*vect[:,ii]*vect[:,jj]*vect[:,kk]/Lvect**2 ).reshape(nx,ny,nz)
                              FUNmom               = FUN_savemom * (vect[:,ii]*vect[:,ii]*vect[:,jj]*vect[:,kk]/Lvect**2 ).reshape(nx,ny,nz)                                   
        #8.2: calculate FFT
                          FUN_out = scipy.fft.fftn( FUN )[:,:,:nz//2].flatten()
                          if((PS_type=='crs')and(PS_multi=='all')): FUN_outmom = scipy.fft.fftn( FUNmom )[:,:,:nz//2].flatten()  
                          if (PS_type  != 'crs'):
                              tmp1      = kvect[:,ii]*kvect[:,ii]*kvect[:,jj]*kvect[:,kk]*((np.real(FUN_out) * np.real(FUN_out_save) + np.imag(FUN_out) * np.imag(FUN_out_save) ) * grid_cor*grid_cor) 
                          if ((PS_type == 'crs')and(PS_multi=='all')):
                              tmp2      = kvect[:,ii]*kvect[:,ii]*kvect[:,jj]*kvect[:,kk]*(0.5*( np.real(FUN_out) * np.imag(FUN_out_savemom) -np.imag(FUN_out) * np.real(FUN_out_savemom) +np.real(FUN_outmom)  * np.imag(FUN_out_save) - np.imag(FUN_outmom) * np.real(FUN_out_save) )* grid_cor*grid_cor)   
        #8.3 calculate l=4:                                         
                          for ikbin in range(nk):                           
                              kprefac = 1.0;
                              if (ii == jj):
                                  if (ii != kk): kprefac = 4.0;
                              else: 
                                  if (jj == kk): kprefac = 6.0;
                                  else :kprefac = 12.0;
                              INDuse=induse[ikbin]; KW=kw[INDuse] ; indkw=(KW>0);    
                              if (PS_type  != 'crs'):
                                  p4[ikbin] = p4[ikbin]+np.sum((4.375*kprefac * tmp1[INDuse])[indkw]/KW[indkw]**4 )
                              if ((PS_type == 'crs')and(PS_multi=='all')):
                                  p4[ikbin] = p4[ikbin]+np.sum((4.375*kprefac * tmp2[INDuse])[indkw]/KW[indkw]**4 )        
                      jusm=jusm+1           
            tmp1=0.;tmp2=0.
    #====================        The end of PS multiples ======================
    #9. save data -------
    # normalize ps: 
    for ikw in range(nk):
        if( Nbink[ikw]>0. ):
            p0[ikw] = p0[ikw]/(Nbink[ikw]*Norm);
            if((PS_multi=='yes')or(PS_multi=='all')):
               p1[ikw] = p1[ikw]*3./(Nbink[ikw]*Norm);
               p2[ikw] = p2[ikw]*5./(Nbink[ikw]*Norm);
               p3[ikw] = p3[ikw]*7./(Nbink[ikw]*Norm);
               p4[ikw] = p4[ikw]*9./(Nbink[ikw]*Norm);    
    #set the last k-bin to be 0:  
    Nbink[-1]=0.
    # save data:
    outfile = open(file_dir, 'w')
    outfile.write("# No.          k            P0            P1           P2           P3           P4            Nk           Norm           SNois  \n") 
    if((PS_multi!='no')):
      for i in range(nk):
        if(i==nk-1):
            outfile.write("  %7d     %17.10lf     %7d     %7d     %7d     %7d     %7d     %7d     %7d     %7d \n"%(i+1, kmin+(i+0.5)*dk,0,0,0,0,0,0,0,0))
        else:
            outfile.write("  %7d     %17.10lf     %17.10lf     %17.10lf     %17.10lf     %17.10lf     %17.10lf    %7d     %30.20lf     %30.20lf \n"%(i+1, kmin+(i+0.5)*dk,np.real(p0[i]),np.real(p1[i]),np.real(p2[i]),np.real(p3[i]),np.real(p4[i]),Nbink[i],Norm,PSN))
    if(PS_multi=='no'):
      for i in range(nk):
        if(i==nk-1):
          outfile.write("  %7d     %17.10lf     %7d     %7d     %7d     %7d     %7d     %7d     %7d     %7d \n"%(i+1, kmin+(i+0.5)*dk,0,0,0,0,0,0,0,0))
        else:
          outfile.write("  %7d     %17.10lf     %17.10lf     %7d     %7d     %7d     %7d     %7d     %30.20lf     %30.20lf \n"%(i+1, kmin+(i+0.5)*dk,np.real(p0[i]),0,0,0,0,Nbink[i],Norm,PSN))
    outfile.close()   
    return  file_dir,Norm   

     
    

             
#########################     The end of Sec 2.    ############################    
###############################################################################      
    