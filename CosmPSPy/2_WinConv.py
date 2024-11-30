import matplotlib.pyplot as plt
import numpy as np
from PSmodFun import *
import gc 




OmegaM      = 0.3121
OmegaA      = 1.0-OmegaM
Hub         = 100.
sigv        = 300.


WFCon_type  = 'Blake'  #  'Ross'     #  'Beutler' #
PS_type     = 'crs'      # 'den'      #  'mom'    #           
PS_measure  = 'mocks'  #'survey' #     



nx,ny,nz    = 128,128,128
lx,ly,lz    = 650.0, 650.0, 650.0
Ncosa       = 200                    # intigration precision for Ross method only.
epsilon_par = [0.0,0.8,2000]         # intigration precision and intervel for Ross method only.
kmin ,kmax ,nkbin  = 0.  , 0.4 , 40  # k_data bins: should be the same as the PSestFun.f90 code.
kminc,kmaxc,nkbinc = 0.  , 0.45, 270 # k_mod bins: the kbin of model PS.
FKP,  FKPv  = 1600 , 5.0e9 
NczBin      = 50
frac_nyq    = 0.9                    # this is for Beutler method only.
if(PS_measure  == 'survey' ):
    randg_Ratio, randv_Ratio  =  116.59168901525683, 1688.2354530106923 
    WeitRg_Ratio,WeitRv_Ratio =  randg_Ratio, randv_Ratio            # this is for Beutler method only.
    Pkden_norm ,Pkmom_norm ,Pkcrs_norm = 9.3785, 3.2509*10**(-13), 1.7461*10**(-6) # this is for Beutler method only.  
if(PS_measure  == 'mocks' ):
    randg_Ratio, randv_Ratio  =  110.16111933176775, 1688.2354530106927
    WeitRg_Ratio,WeitRv_Ratio =  randg_Ratio, randv_Ratio            # this is for Beutler method only.
    Pkden_norm ,Pkmom_norm ,Pkcrs_norm = 9.4082, 3.2575*10**(-13), 1.7505*10**(-6) # this is for Beutler method only. they are average of the 600 mocks. 
    
    
file_dir      = '/Users/fei/WSP/Scie/CosmPS/Data/ProdPy/'  
fileOri_dir   = '/Users/fei/WSP/Scie/CosmPS/Data/Orig/'  
if(WFCon_type ==  'Beutler' ): 
    fileOri_dir   = '/global/homes/f/feiqin/Scie/Proj12/Data/Orig/'
    file_dir  = '/global/homes/f/feiqin/Scie/Proj12/Data/Orig/'













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
    wingridgal,weigridgal = Prep_winFUN(randg_Ratio,FKP,randg_x,randg_y,randg_z,np.nan,sigv,nx,ny,nz,lx,ly,lz,'den')
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
    wingridpv,weigridpv   = Prep_winFUN(randv_Ratio,FKPv,randv_x,randv_y,randv_z,vpR,sigv,nx,ny,nz,lx,ly,lz,'mom')
 




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
if(WFCon_type  == 'Ross'):
 plt.figure(2)
 plt.pcolor(WFij)
 plt.title('den WF-convolution matrix')
 plt.xlabel('$k_j$ model')
 plt.ylabel('$k_i$ data')
 plt.show()
else:
 plt.figure(3)
 plt.pcolor(conv )
 plt.title('den WF-convolution matrix')
 plt.xlabel('$k_j$ model')
 plt.ylabel('$k_i$ data') 
 plt.show()
 plt.imshow(weigridgal[:,:,nz//3]);plt.show()
 plt.imshow(weigridpv[:,:,nz//3]);plt.show()
   #'''
