import matplotlib.pyplot as plt
import numpy as np
from PSmodFun import *
import gc 




OmegaM      = 0.3
OmegaA      = 1.0-OmegaM
Hub         = 100.
sigv        = 300.


WFCon_type  = 'Ross'     #'Blake'  #    'Beutler' #
PS_type     =   'den'    # 'mom'    #    'crs'      #       
PS_measure  =  'mocks'  #'survey' #    



nx,ny,nz    = 128,128,128
lx,ly,lz    = 650.0, 650.0, 650.0
Ncosa       = 200                    # intigration precision for Ross method only.
epsilon_par = [0.0,0.8,2000]         # intigration precision and intervel for Ross method only.
kmin ,kmax ,nkbin  = 0.  , 0.4 , 40  # k_data bins: should be the same as the PSestFun.f90 code.
kminc,kmaxc,nkbinc = 0.  , 0.42, 270 # k_mod bins: the kbin of model PS.
FKP,  FKPv  = 1600 , 5.0e9 
NczBin      = 50
frac_nyq    = 0.9                    # this is for Beutler method only.
if(PS_measure  == 'survey' ):
    randg_Ratio, randv_Ratio  =  116.59168901525683,  1688.2354530106923
    WeitRg_Ratio,WeitRv_Ratio =  112.75647879064488,  1657.2744260916527     # this is for Beutler method only.
    Pkden_norm ,Pkmom_norm ,Pkcrs_norm = 9.37848, 3.25087*10**(-13), 1.746089*10**(-6) # this is for Beutler method only.  
if(PS_measure  == 'mocks' ):
    randg_Ratio, randv_Ratio  =  110.16111933176775, 1688.2354530106927
    WeitRg_Ratio,WeitRv_Ratio =  114.50254965704735, 1655.5829495778107     # this is for Beutler method only.
    Pkden_norm ,Pkmom_norm ,Pkcrs_norm = 9.40913, 3.25740*10**(-13), 1.750615*10**(-6) # this is for Beutler method only. they are average of the 600 mocks. 
    
    
file_dir      = '/Users/fei/WSP/Scie/FitExamp/Data/ProdPy/'  
fileOri_dir   = '/Users/fei/WSP/Scie/FitExamp/Data/Orig/'  
if(WFCon_type ==  'Beutler' ): 
    fileOri_dir   = "../Data/"
    file_dir  = "../Data/"













print('\n 1.Note: The nx,ny,nz,nkbinc should not too large,')
print('unless the number of random points are large enough')
print('to produce a smooth grid for window function grid. ')
print('\n 2.Note: The kmaxc should be slightly larger then max(k_data),')
print('but if kmaxc is much larger then max(k_data), the convolved PS')
print('will has a shift compare to the unconvolved PS. So do not set kmaxc too large.')
print('\n 3.Note: The randg_Ratio of the mocks is different from the randg_Ratio of')
print('real data, so you may want to calculate convmat for mocks and real data seperatly.')
print('So does randv_Ratio. In this code we only use the Ratio of mocks. \n')
print('convmat-',PS_type,'  method-',WFCon_type,'  catlog-',PS_measure,'\n')
# 1. read randon data : ------------------------------------------------------- 
if(WFCon_type!='Ross'):  
    infile   =np.load(fileOri_dir+'Rand.npy',allow_pickle=True)
    raR      = infile[0]
    decR     = infile[1]
    czR      = infile[2] 
    nbR      = infile[3]  
    nbR      = nbR / randg_Ratio
    randg_weit=  1./(   1.+nbR*FKP)
    randg_x,randg_y,randg_z  = Sky2Cat(raR,decR,czR/LightSpeed,OmegaM , OmegaA ,Hub,nbin=20000,redmax=2)   
    del infile,raR,decR,czR,nbR
    gc.collect()
    wingridgal,weigridgal = Prep_winFUN(randg_Ratio,FKP,randg_x,randg_y,randg_z,np.nan,1.,np.nan,nx,ny,nz,lx,ly,lz,'den')
    #  PV  :------------------ 
    infile   = np.load(fileOri_dir+'Datav.npy',allow_pickle=True)
    czv      = infile[2] 
    logd     = infile[3]
    infile   = np.load(fileOri_dir+'Randv.npy',allow_pickle=True)
    ravR     = infile[0]
    decvR    = infile[1]
    czvR     = infile[2] 
    nbvR     = infile[3] 
    nbvR     = nbvR / randv_Ratio
    vpR,evpR = VpRand_Fun(logd,czv,czvR,NczBin,OmegaM)
    randv_weit= 1./(evpR**2+sigv**2+ nbvR*FKPv) 
    randv_x,randv_y,randv_z  = Sky2Cat(ravR,decvR,czvR/LightSpeed,OmegaM , OmegaA ,Hub,nbin=20000,redmax=2)
    del infile,ravR,decvR,czvR,nbvR 
    gc.collect()
    wingridpv,weigridpv   = Prep_winFUN(randv_Ratio,FKPv,randv_x,randv_y,randv_z,evpR,1.,sigv,nx,ny,nz,lx,ly,lz,'mom')




# 2. calculate convolution matrix :--------------------------------------------
# 2.1 Ross method:
if(WFCon_type  == 'Ross'):
    # PS of rand for Ross: 
    infile=np.loadtxt(file_dir+'Data_'+PS_type)
    k_data=infile[:,1]
    NKs= len(k_data)
    infile=np.loadtxt(file_dir+'Rand_'+PS_type)
    k_rand=infile[0:len(infile[:,0])-1,1];Pk_rand=infile[0:len(infile[:,0])-1,2];k_rand[0]=0.0;Pk_rand[0]=1.0
    ind=np.where(Pk_rand>0.0);k_rand =k_rand[ind];Pk_rand =Pk_rand[ind]
    ind=np.where(Pk_rand>0.0);k_rand =k_rand[ind];Pk_rand =Pk_rand[ind]
    # extend the ps of randoms:
    plt.figure(1,figsize=(9.5,7))
    plt.plot(np.log10(k_rand), np.log10(Pk_rand),marker='',color='gray')
    ktemp,Pktemp=PS_extender(k_rand[1:len(k_rand)],Pk_rand[1:len(k_rand)],100,len(k_rand))
    k_rands =np.zeros(1+len(ktemp));Pk_rands=np.zeros(1+len(ktemp))
    k_rands[0]=0.0;Pk_rands[0]=1.0
    k_rands[1:len(k_rands)]=ktemp;Pk_rands[1:len(k_rands)]=Pktemp
    k_rand=k_rands
    Pk_rand=Pk_rands
    ind=np.where(Pk_rand>0.0)
    k_rand =k_rand[ind]
    Pk_rand =Pk_rand[ind]
    plt.plot(np.log10(k_rand), np.log10(Pk_rand),marker='',color='r') 
    plt.show()     
    # the matrix WF_CONV[ki][kj] is used for convert the 
    # model PS, Pt[j] to match the measured PS, Pm[i].
    # 2.1: Ki is for the output, which corresponds to measured PS:
    Ki = k_data[0:NKs-1]
    WF_rand =  [k_rand,Pk_rand] 
    # 2.4: The WF matrix WFij is to convert the input model PS(kj) 
    # to the output PS(ki):
    WFij,Kj=ConvMat_Ross(Ki,np.min(Ki),kmaxc,nkbinc,WF_rand,Ncosa,epsilon_par) # win[0:39,:]#
    np.save(file_dir+'conv'+PS_type+'_'+WFCon_type+'_'+PS_measure,np.array([WFij,Ki,Kj,WF_rand],dtype=object),allow_pickle=True)  
 
# 2.2 Black method:--------------
if(WFCon_type == 'Blake')  :
    if(PS_type=='den'):
        conv,kd,kc   =  ConvMat_Blake(kmin,kmax,nkbin,kmaxc,nkbinc,nx,ny,nz,lx,ly,lz,weigridgal,wingridgal ,weigridgal,wingridgal,PS_type) 
    if(PS_type=='mom'):
        conv,kd,kc   =  ConvMat_Blake(kmin,kmax,nkbin,kmaxc,nkbinc,nx,ny,nz,lx,ly,lz,weigridpv ,wingridpv  ,weigridpv ,wingridpv ,PS_type) 
    if(PS_type=='crs'):
        conv,kd,kc   =  ConvMat_Blake(kmin,kmax,nkbin,kmaxc,nkbinc,nx,ny,nz,lx,ly,lz,weigridgal,wingridgal ,weigridpv ,wingridpv ,PS_type) 
    np.save(file_dir+'conv'+PS_type+'_'+WFCon_type+'_'+PS_measure,np.array([conv,kd,kc],dtype=object),allow_pickle=True)
     
# 2.3 pypower method:----------- 
if(WFCon_type == 'Beutler'):
    boxsize,boxsizes=3*lx,lx
    Ngrid=nx
    nkbinc=6300#9*nkbinc
    if(PS_type=='den'):
        conv,kd,kc   = ConvMat_Beutler( randg_x ,randg_y,randg_z,randg_weit,Ngrid,Pkden_norm,WeitRg_Ratio,kmin,kmax,nkbin,kminc,kmaxc,nkbinc,[0,2,4],boxsize,boxsizes,frac_nyq, PS_type,WFCon_type)
    if(PS_type=='mom'):
        conv,kd,kc   = ConvMat_Beutler( randv_x ,randv_y,randv_z,randv_weit,Ngrid,Pkmom_norm,WeitRv_Ratio,kmin,kmax,nkbin,kminc,kmaxc,nkbinc,[0,2,4],boxsize,boxsizes,frac_nyq, PS_type,WFCon_type)
    if(PS_type=='crs'):
        NrandRat=[WeitRg_Ratio,WeitRv_Ratio]
        Xr      =[randg_x     ,randv_x] 
        Yr      =[randg_y     ,randv_y]
        Zr      =[randg_z     ,randv_z]
        weitR   =[randg_weit  ,randv_weit ]
        conv,kd,kc   =  ConvMat_Beutler( Xr,Yr,Zr, weitR,Ngrid,Pkcrs_norm,NrandRat,kmin,kmax,nkbin,kminc,kmaxc,nkbinc, [1,3],boxsize,boxsizes,frac_nyq, PS_type,WFCon_type)
    np.save(file_dir+'conv'+PS_type+'_'+WFCon_type+'_'+PS_measure,np.array([conv,kd,kc],dtype=object),allow_pickle=True)











# 3. make plots:---------------------------------------------------------------
if(WFCon_type != 'Ross'):
  conv,kd,kc   =np.load(file_dir+'conv'+PS_type+'_'+WFCon_type+'_'+PS_measure+'.npy',allow_pickle=True)
  if(PS_type != 'crs'   ):
    plt.figure(3,figsize=(9,7))
    Kd=np.concatenate((kd,kd,kd)) ;Nkd=len(kd)//3; indkd=np.array([Nkd,2*Nkd,3*Nkd,len(kd)+Nkd,len(kd)+2*Nkd,len(kd)+3*Nkd,2*len(kd)+Nkd,2*len(kd)+2*Nkd,2*len(kd)+3*Nkd],dtype=int)-Nkd//2    
    kcut=0.01;Kc0=np.concatenate((kc,kc,kc));kc=kc[kc>=kcut];Kc=np.concatenate((kc,kc,kc)) ;Nkc=len(kc)//3; indkc=np.array([Nkc,2*Nkc,3*Nkc,len(kc)+Nkc,len(kc)+2*Nkc,len(kc)+3*Nkc,2*len(kc)+Nkc,2*len(kc)+2*Nkc,2*len(kc)+3*Nkc],dtype=int)-Nkc//2        
    ind=np.where(Kc0>=kcut)[0];convs=conv[:,ind]        
    fsiz=25;tsiz=15
    im   = plt.pcolor( convs  ,    cmap='RdBu_r'   )
    cbar = plt.colorbar() ;plt.clim(0.8*np.min(conv),  0.7*np.max(conv));
    cbar.ax.tick_params(labelsize=tsiz); cbar.set_label('$W(k_{mod},k_{mea})$',fontsize=fsiz)  ;
    plt.tick_params(axis = 'both', which = 'major', labelsize =tsiz); 
    plt.xlabel('$k_{mod}$ [h Mpc$^{-1}$]',fontsize=fsiz);
    plt.ylabel('$k_{mea}$ [h Mpc$^{-1}$]',fontsize=fsiz) ;
    plt.plot([0,3*len(kc)],[len(kd),len(kd)],c='w',ls='-.',lw=0.8)
    plt.plot([0,3*len(kc)],[2*len(kd),2*len(kd)],c='w',ls='-.',lw=0.8)
    plt.plot([0,3*len(kc)],[3*len(kd),3*len(kd)],c='w',ls='-.',lw=0.8)
    plt.plot([len(kc),len(kc)],[0,3*len(kd)],c='w',ls='-.',lw=0.8)
    plt.plot([2*len(kc),2*len(kc)],[0,3*len(kd)],c='w',ls='-.',lw=0.8)
    plt.plot([3*len(kc),3*len(kc)],[0,3*len(kd)],c='w',ls='-.',lw=0.8)
    plt.xticks(indkc,np.round(Kc[indkc],3));
    plt.yticks(indkd,np.round(Kd[indkd],3));
  if(PS_type == 'crs'   ):
    plt.figure(3,figsize=(9,7))
    Kd=np.concatenate((kd,kd)) ;Nkd=len(kd)//3; indkd=np.array([Nkd,2*Nkd,3*Nkd,len(kd)+Nkd,len(kd)+2*Nkd,len(kd)+3*Nkd],dtype=int)-Nkd//2    
    kcut=0.01;Kc0=np.concatenate((kc,kc));kc=kc[kc>=kcut];Kc=np.concatenate((kc,kc)) ;Nkc=len(kc)//3; indkc=np.array([Nkc,2*Nkc,3*Nkc,len(kc)+Nkc,len(kc)+2*Nkc,len(kc)+3*Nkc],dtype=int)-Nkc//2        
    ind=np.where(Kc0>=kcut)[0];convs=conv[:,ind]        
    fsiz=25;tsiz=15
    im   = plt.pcolor( convs  ,    cmap='RdBu_r'   )
    cbar = plt.colorbar() ;plt.clim(0.8*np.min(conv),  0.7*np.max(conv));
    cbar.ax.tick_params(labelsize=tsiz); cbar.set_label('$W(k_{mod},k_{mea})$',fontsize=fsiz)  ;
    plt.tick_params(axis = 'both', which = 'major', labelsize =tsiz); 
    plt.xlabel('$k_{mod}$ [h Mpc$^{-1}$]',fontsize=fsiz);
    plt.ylabel('$k_{mea}$ [h Mpc$^{-1}$]',fontsize=fsiz) ;
    plt.plot([0,2*len(kc)],[len(kd),len(kd)],c='w',ls='-.',lw=0.8)
    plt.plot([0,2*len(kc)],[2*len(kd),2*len(kd)],c='w',ls='-.',lw=0.8)
    plt.plot([len(kc),len(kc)],[0,2*len(kd)],c='w',ls='-.',lw=0.8)
    plt.plot([2*len(kc),2*len(kc)],[0,2*len(kd)],c='w',ls='-.',lw=0.8)
    plt.xticks(indkc,np.round(Kc[indkc],3));
    plt.yticks(indkd,np.round(Kd[indkd],3));
else:
    conv,kd,kc,tmp   =np.load(file_dir+'conv'+PS_type+'_'+WFCon_type+'_'+PS_measure+'.npy',allow_pickle=True)
    Nkd=len(kd)//4; indkd=np.array([0.5*Nkd,Nkd,1.5*Nkd,2*Nkd,2.5*Nkd,3*Nkd,3.5*Nkd,4*Nkd],dtype=int)-1        
    Nkc=len(kc)//4; indkc=np.array([0.5*Nkc,Nkc,1.5*Nkc,2*Nkc,2.5*Nkc,3*Nkc,3.5*Nkc,4*Nkc],dtype=int)             
    plt.figure(3,figsize=(6,5))
    fsiz=18;tsiz=13
    im   = plt.pcolor( conv   ,    cmap='RdBu_r'   )
    cbar = plt.colorbar() ;plt.clim(np.min(conv),  np.max(conv));
    cbar.ax.tick_params(labelsize=tsiz); cbar.set_label('$W(k_{mod},k_{mea})$',fontsize=fsiz)  ;
    plt.tick_params(axis = 'both', which = 'major', labelsize =tsiz); 
    plt.xlabel('$k_{mod}$ [h Mpc$^{-1}$]',fontsize=fsiz);
    plt.ylabel('$k_{mea}$ [h Mpc$^{-1}$]',fontsize=fsiz) ;
    plt.xticks(indkc,np.round(kc[indkc],3));
    plt.yticks(indkd,np.round(kd[indkd],3));
    plt.show()         
    