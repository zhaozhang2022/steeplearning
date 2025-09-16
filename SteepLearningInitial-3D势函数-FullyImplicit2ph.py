import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# 倒入计算逆矩阵的函数inv()
from numpy.linalg import inv
from numpy.linalg import cholesky, det, lstsq
from scipy.optimize import minimize
from scipy.signal import ricker
from scipy.sparse.linalg import cg
import torch
import sys
import torchvision
import torch.utils.data
import math
import time
import scipy
from torch.autograd.functional import jacobian
#全隐式-系数隐式-流匹配-mse loss反向
class NODE:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0

class CELL:
    def __init__(self): # 不加self就变成了对所有类对象同时更改
        self.vertices = [-1, -1, -1, -1, -1, -1, -1, -1]
        self.neighbors = [-1, -1, -1, -1, -1, -1]
        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.volume = 0
        self.xc = 0
        self.yc = 0
        self.zc = 0
        self.porosity = 0
        self.kx = 0
        self.ky = 0
        self.kz = 0
        self.trans = [0, 0, 0, 0, 0, 0]
        self.markbc = 0
        self.markwell = 0

class FLUID:
    def __init__(self):
        self.mur=0
        self.cmu=0
        self.cf=0 #compressibility
        self.rhor=0 #reference rho
        self.pr=0 #reference press

class ROCK:
    def __init__(self):
        self.cr=0 #compressibility
        self.poror = 0 # reference porosity
        self.pr = 0  # reference press

class WELL:
    def __init__(self):
        self.pwf=0
        self.PI=0
        self.q=0

class CallingCounter(object):
    def __init__ (self, func):
        self.func = func
        self.count = 0

    def __call__ (self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)

def BuildCartesianGrid(nx,ny,nz,dx,dy,dz):
    print("build Grid")
    dxvec = [0]
    for i in range(0, nx):
        dxvec.append(dx)

    dyvec = [0]
    for i in range(0, ny):
        dyvec.append(dy)
    dzvec = [0]
    for i in range(0, nz):
        dzvec.append(dz)

    nx = len(dxvec) - 1
    ny = len(dyvec) - 1
    nz = len(dzvec) - 1
    nodelist = []
    llz = 0
    for k in range(0, nz + 1):
        llz = llz + dzvec[k]
        lly = 0
        for j in range(0, ny + 1):
            lly = lly + dyvec[j]
            llx = 0
            for i in range(0, nx + 1):
                llx = llx + dxvec[i]
                node = NODE()
                node.x = llx
                node.y = lly
                node.z = llz
                nodelist.append(node)

    # build connectivity and neighbors
    celllist = []

    for k in range(0, nz):
        for j in range(0, ny):
            for i in range(0, nx):
                id = k * nx * ny + j * nx + i
                nc = id
                cell = CELL()
                if i > 0:
                    cell.neighbors[0] = nc - 1
                if i < nx - 1:
                    cell.neighbors[1] = nc + 1
                if j > 0:
                    cell.neighbors[2] = nc - nx
                if j < ny - 1:
                    cell.neighbors[3] = nc + nx
                if k > 0:
                    cell.neighbors[4] = nc - nx * ny
                if k < nz - 1:
                    cell.neighbors[5] = nc + nx * ny
                i0 = k * (nx + 1) * (ny + 1) + j * (nx + 1) + i
                i1 = k * (nx + 1) * (ny + 1) + j * (nx + 1) + i + 1
                i2 = k * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i
                i3 = k * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i + 1
                i4 = (k + 1) * (nx + 1) * (ny + 1) + j * (nx + 1) + i
                i5 = (k + 1) * (nx + 1) * (ny + 1) + j * (nx + 1) + i + 1
                i6 = (k + 1) * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i
                i7 = (k + 1) * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i + 1
                cell.dx = nodelist[i1].x - nodelist[i0].x
                cell.dy = nodelist[i2].y - nodelist[i0].y
                cell.dz = nodelist[i4].z - nodelist[i0].z
                cell.vertices[0] = i0
                cell.vertices[1] = i1
                cell.vertices[2] = i2
                cell.vertices[3] = i3
                cell.vertices[4] = i4
                cell.vertices[5] = i5
                cell.vertices[6] = i6
                cell.vertices[7] = i7
                cell.xc = 0.125 * (nodelist[i0].x + nodelist[i1].x + nodelist[i2].x + nodelist[i3].x + nodelist[i4].x +
                                   nodelist[i5].x + nodelist[i6].x + nodelist[i7].x)
                cell.yc = 0.125 * (nodelist[i0].y + nodelist[i1].y + nodelist[i2].y + nodelist[i3].y + nodelist[i4].y +
                                   nodelist[i5].y + nodelist[i6].y + nodelist[i7].y)
                cell.zc = 0.125 * (nodelist[i0].z + nodelist[i1].z + nodelist[i2].z + nodelist[i3].z + nodelist[i4].z +
                                   nodelist[i5].z + nodelist[i6].z + nodelist[i7].z)
                cell.volume = cell.dx * cell.dy * cell.dz
                celllist.append(cell)
    return celllist, nodelist


def SetRockFluid2ph():
    print("define rock and fluid properties")
    rock = ROCK()
    rock.poror = 0.1
    rock.pr = 20e6
    rock.cr = 1e-11

    fluidn = FLUID()
    fluidn.pr = 20e6
    fluidn.rhor = 850.0
    fluidn.cf = 5e-8
    fluidn.mur = 2e-3
    fluidn.cmu = 1e-18

    fluidw = FLUID()
    fluidw.pr = 20e6
    fluidw.rhor = 1000.0
    fluidw.cf = 5e-9
    fluidw.mur = 1e-3
    fluidw.cmu = 1e-18
    return rock, fluidn, fluidw

def Rho(p,pr,cf,rhor):
    rho_=rhor*torch.exp(cf*(p-pr))
    return rho_

def Poro(p,pr,cr,poror):
    poro_=poror*torch.exp(cr*(p-pr))
    return poro_

def Mu(p,pr,cmu,mur):
    mu_=mur*(1+cmu*(p-pr))
    return mu_

def Krw(sw):
    krw0=0.5
    swr=0.2
    snr=0.2
    nw=2.0
    krw_=krw0*((sw-swr)/(1.0-swr-snr))**nw
    return krw_
def Krn(sn):
    krn0=1.0
    swr=0.2
    snr=0.2
    nn=2.0
    krn_=krn0*((sn-snr)/(1.0-swr-snr))**nn
    return krn_

def Pc(sw):
    swr = 0.2
    snr = 0.2
    pe=10000
    pc_=pe*((1.0-sw-snr)/(1.0-swr-snr))
    return pc_

def SetWell():
    rw = 0.05
    SS = 3
    # length = 3000
    # cs = fluid.cf * 3.14 * rw * rw * length
    ddx = 15
    re = 0.14 * (ddx * ddx + ddx * ddx) ** 0.5
    PI = 2 * 3.14 * ddx * 2.5e-15 / (math.log(re / rw) + SS)
    pwf = 15e6  # bottom-hole pressure
    qwt = 20.0/86400
    for k in range(0, nz):
        for j in range(0, ny):
            for i in range(0, nx):
                id = k * nx * ny + j * nx + i
                if i == 0 and j == 0:
                    celllist[id].markwell = 1
                    celllist[id].markbc = -2  # 生产井
                elif i == nx - 1 and j == 0:
                    celllist[id].markwell = 2
                    celllist[id].markbc = -2  # 生产井
                elif i == 0 and j == ny - 1:
                    celllist[id].markwell = 3
                    celllist[id].markbc = -2  # 生产井
                elif i == nx - 1 and j == ny - 1:
                    celllist[id].markwell = 4
                    celllist[id].markbc = -2  # 生产井
                elif i == nx / 2 and j == ny / 2:
                    celllist[id].markwell = 5
                    celllist[id].markbc = -1  # 注水井
    return PI, pwf, qwt


def Buildtrans(chukvec):
    for i in range(0, ncell):  # set chuk
        celllist[i].kx = chukvec[i]
        celllist[i].ky = chukvec[i]
        celllist[i].kz = chukvec[i]*0.1
    for ie in range(0, ncell):  # compute transmissibility
        dx1 = celllist[ie].dx
        dy1 = celllist[ie].dy
        dz1 = celllist[ie].dz
        for j in range(0, 6):
            je = celllist[ie].neighbors[j]
            if je >= 0:
                dx2 = celllist[je].dx
                dy2 = celllist[je].dy
                dz2 = celllist[je].dz
                mt1 = 1.0 # 不包含mu
                mt2 = 1.0
                if j == 0 or j == 1:
                    mt1 = mt1 * dy1 * dz1
                    mt2 = mt2 * dy2 * dz2
                    k1 = celllist[ie].kx
                    k2 = celllist[je].kx
                    dd1 = dx1 / 2.
                    dd2 = dx2 / 2.
                elif j == 2 or j == 3:
                    mt1 = mt1 * dx1 * dz1
                    mt2 = mt2 * dx2 * dz2
                    k1 = celllist[ie].ky
                    k2 = celllist[je].ky
                    dd1 = dy1 / 2.
                    dd2 = dy2 / 2.
                elif j == 4 or j == 5:
                    mt1 = mt1 * dx1 * dy1
                    mt2 = mt2 * dx2 * dy2
                    k1 = celllist[ie].kz
                    k2 = celllist[je].kz
                    dd1 = dz1 / 2.
                    dd2 = dz2 / 2.
                t1 = mt1 * k1 / dd1
                t2 = mt2 * k2 / dd2
                tt = 1 / (1 / t1 + 1 / t2)
                celllist[ie].trans[j] = tt


def Buildneibortensor():
    neibortensor_0= torch.zeros((ncell)).type(torch.long)
    neibortensor_1 = torch.zeros((ncell)).type(torch.long)
    neibortensor_2 = torch.zeros((ncell)).type(torch.long)
    neibortensor_3 = torch.zeros((ncell)).type(torch.long)
    neibortensor_4 = torch.zeros((ncell)).type(torch.long)
    neibortensor_5 = torch.zeros((ncell)).type(torch.long)
    for ie in range(ncell):
        neibors = celllist[ie].neighbors
        for j in range(6):
            if neibors[j]<0:
                neibors[j]=0
        neibortensor_0[ie]=torch.tensor(neibors[0],dtype=torch.int32)
        neibortensor_1[ie] = torch.tensor(neibors[1], dtype=torch.int32)
        neibortensor_2[ie] = torch.tensor(neibors[2], dtype=torch.int32)
        neibortensor_3[ie] = torch.tensor(neibors[3], dtype=torch.int32)
        neibortensor_4[ie] = torch.tensor(neibors[4], dtype=torch.int32)
        neibortensor_5[ie] = torch.tensor(neibors[5], dtype=torch.int32)
    return neibortensor_0, neibortensor_1,neibortensor_2,neibortensor_3, neibortensor_4,neibortensor_5

def Buildtranstensor():
    transvec_0 = torch.zeros(ncell, dtype=torch.float64)
    transvec_1 = torch.zeros(ncell, dtype=torch.float64)
    transvec_2 = torch.zeros(ncell, dtype=torch.float64)
    transvec_3 = torch.zeros(ncell, dtype=torch.float64)
    transvec_4 = torch.zeros(ncell, dtype=torch.float64)
    transvec_5 = torch.zeros(ncell, dtype=torch.float64)
    for ie in range(ncell):
        transvec_0[ie] = celllist[ie].trans[0]
        transvec_1[ie] = celllist[ie].trans[1]
        transvec_2[ie] = celllist[ie].trans[2]
        transvec_3[ie] = celllist[ie].trans[3]
        transvec_4[ie] = celllist[ie].trans[4]
        transvec_5[ie] = celllist[ie].trans[5]
    return transvec_0, transvec_1, transvec_2, transvec_3, transvec_4, transvec_5

def Buildztensor():
    zvec=torch.zeros(ncell, dtype=torch.float64)
    for ie in range(ncell):
        zvec[ie]=celllist[ie].zc
    return zvec

def Buildmarkbcvec():
    markbcvec=torch.zeros(ncell,dtype=torch.int32)
    for ie in range(ncell):
        markbcvec[ie]=celllist[ie].markbc
    return markbcvec


def Resw(pvec1, pvec0,swvec1, swvec0,fluidw): #residual for the ie function of wetting phase
    # pore volume change
    pcvec0=Pc(swvec0)
    pcvec1 = Pc(swvec1)
    pvec0 = pvec0-pcvec0 #wetting pressure
    pvec1 = pvec1-pcvec1
    porovec0 = Poro(pvec0+gvec, rock.pr, rock.cr, rock.poror)
    porovec1 = Poro(pvec1+gvec, rock.pr, rock.cr, rock.poror)
    rhovec0 = Rho(pvec0+gvec, fluidw.pr, fluidw.cf, fluidw.rhor)
    rhovec1 = Rho(pvec1+gvec, fluidw.pr, fluidw.cf, fluidw.rhor)
    pvvec_ = cellvolume_ / dt * (swvec1*porovec1 * rhovec1 - swvec0*porovec0 * rhovec0)
    # source
    krwvec1=Krw(swvec1) #relative perm
    muvec1=Mu(pvec1+gvec,fluidw.pr,fluidw.cmu, fluidw.mur)

    mobivec1=krwvec1/muvec1
    sourcevec_=torch.where(markbcvec==-2, -rhovec1*mobivec1*PI * (pvec0 - pwf), 0)
    sourcevec_=torch.where(markbcvec==-1, rhovec1*qwfix, sourcevec_)

    # flux
    rhoavvec_0 = (rhovec1 + rhovec1[neibortensor_0])*0.5
    rhoavvec_1 = (rhovec1 + rhovec1[neibortensor_1])*0.5
    rhoavvec_2 = (rhovec1 + rhovec1[neibortensor_2])*0.5
    rhoavvec_3 = (rhovec1 + rhovec1[neibortensor_3])*0.5
    rhoavvec_4 = (rhovec1 + rhovec1[neibortensor_4])*0.5
    rhoavvec_5 = (rhovec1 + rhovec1[neibortensor_5])*0.5
    mobiupvec_0 = torch.where(pvec1 >= pvec1[neibortensor_0], mobivec1, mobivec1[neibortensor_0])
    mobiupvec_1 = torch.where(pvec1 >= pvec1[neibortensor_1], mobivec1, mobivec1[neibortensor_1])
    mobiupvec_2 = torch.where(pvec1 >= pvec1[neibortensor_2], mobivec1, mobivec1[neibortensor_2])
    mobiupvec_3 = torch.where(pvec1 >= pvec1[neibortensor_3], mobivec1, mobivec1[neibortensor_3])
    mobiupvec_4 = torch.where(pvec1 >= pvec1[neibortensor_4], mobivec1, mobivec1[neibortensor_4])
    mobiupvec_5 = torch.where(pvec1 >= pvec1[neibortensor_5], mobivec1, mobivec1[neibortensor_5])

    fluxvec_0 = rhoavvec_0 * mobiupvec_0 * transvec_0 * (pvec1[neibortensor_0] - pvec1)
    fluxvec_1 = rhoavvec_1 * mobiupvec_1 * transvec_1 * (pvec1[neibortensor_1] - pvec1)
    fluxvec_2 = rhoavvec_2 * mobiupvec_2 * transvec_2 * (pvec1[neibortensor_2] - pvec1)
    fluxvec_3 = rhoavvec_3 * mobiupvec_3 * transvec_3 * (pvec1[neibortensor_3] - pvec1)
    fluxvec_4 = rhoavvec_4 * mobiupvec_4 * transvec_4 * (pvec1[neibortensor_4] - pvec1)
    fluxvec_5 = rhoavvec_5 * mobiupvec_5 * transvec_5 * (pvec1[neibortensor_5] - pvec1)

    fluxvec_=-1*fluxvec_0-fluxvec_1-fluxvec_2-fluxvec_3-fluxvec_4-fluxvec_5
    resvec_= pvvec_+fluxvec_-sourcevec_
    return resvec_

def Resn(pvec1, pvec0,swvec1, swvec0, fluidn): #residual for the ie function of nonwetting phase
    snvec1=1.0-swvec1
    snvec0=1.0-swvec0
    porovec0 = Poro(pvec0+gvec, rock.pr, rock.cr, rock.poror)
    porovec1 = Poro(pvec1+gvec, rock.pr, rock.cr, rock.poror)
    rhovec0 = Rho(pvec0+gvec, fluidn.pr, fluidn.cf, fluidn.rhor)
    rhovec1 = Rho(pvec1+gvec, fluidn.pr, fluidn.cf, fluidn.rhor)
    pvvec_ = cellvolume_ / dt * (snvec1 * porovec1 * rhovec1 - snvec0 * porovec0 * rhovec0)
    # source
    krnvec1 = Krn(snvec1)  # relative perm
    muvec1 = Mu(pvec1+gvec, fluidn.pr, fluidn.cmu, fluidn.mur)

    mobivec1 = krnvec1/muvec1
    sourcevec_ = torch.where(markbcvec == -2, -rhovec1 * mobivec1 * PI * (pvec0 - pwf), 0)
    # sourcevec_ = torch.where(markbcvec == -1, rhovec1 * qwfix, sourcevec_)

    # flux
    rhoavvec_0 = (rhovec1 + rhovec1[neibortensor_0])*0.5
    rhoavvec_1 = (rhovec1 + rhovec1[neibortensor_1])*0.5
    rhoavvec_2 = (rhovec1 + rhovec1[neibortensor_2])*0.5
    rhoavvec_3 = (rhovec1 + rhovec1[neibortensor_3])*0.5
    rhoavvec_4 = (rhovec1 + rhovec1[neibortensor_4])*0.5
    rhoavvec_5 = (rhovec1 + rhovec1[neibortensor_5])*0.5
    mobiupvec_0 = torch.where(pvec1 >= pvec1[neibortensor_0], mobivec1, mobivec1[neibortensor_0])
    mobiupvec_1 = torch.where(pvec1 >= pvec1[neibortensor_1], mobivec1, mobivec1[neibortensor_1])
    mobiupvec_2 = torch.where(pvec1 >= pvec1[neibortensor_2], mobivec1, mobivec1[neibortensor_2])
    mobiupvec_3 = torch.where(pvec1 >= pvec1[neibortensor_3], mobivec1, mobivec1[neibortensor_3])
    mobiupvec_4 = torch.where(pvec1 >= pvec1[neibortensor_4], mobivec1, mobivec1[neibortensor_4])
    mobiupvec_5 = torch.where(pvec1 >= pvec1[neibortensor_5], mobivec1, mobivec1[neibortensor_5])

    fluxvec_0 = rhoavvec_0 * mobiupvec_0 * transvec_0 * (pvec1[neibortensor_0] - pvec1)
    fluxvec_1 = rhoavvec_1 * mobiupvec_1 * transvec_1 * (pvec1[neibortensor_1] - pvec1)
    fluxvec_2 = rhoavvec_2 * mobiupvec_2 * transvec_2 * (pvec1[neibortensor_2] - pvec1)
    fluxvec_3 = rhoavvec_3 * mobiupvec_3 * transvec_3 * (pvec1[neibortensor_3] - pvec1)
    fluxvec_4 = rhoavvec_4 * mobiupvec_4 * transvec_4 * (pvec1[neibortensor_4] - pvec1)
    fluxvec_5 = rhoavvec_5 * mobiupvec_5 * transvec_5 * (pvec1[neibortensor_5] - pvec1)

    fluxvec_ = -1 * fluxvec_0 - fluxvec_1 - fluxvec_2 - fluxvec_3 - fluxvec_4 - fluxvec_5
    resvec_ = pvvec_ + fluxvec_ - sourcevec_
    return resvec_

def Setinitialcondition():
    pvec = np.ones(ncell)*p_init
    swvec = np.ones(ncell) * 0.2
    return pvec, swvec

def gravityvec():
    pref=0
    zref=27.0 #z points upward which is different from mrst
    rho_=850.0 #用油作为连通相计算静压力
    gravec = np.ones(ncell)
    for ie in range(ncell):
        z_=celllist[ie].zc
        gravec[ie]=rho_*9.8*(zref-z_)+pref
    return gravec

def ComputeInitialGuess(ptensor0, swtensor0, num_epochs, tol):
    lowesttol=1e10
    for epoch in range(num_epochs):
        ptensor1 = predict[0:ncell]*p_init+ptensor0
        swtensor1 = predict[ncell:]+swtensor0
        resvec1 = Resw(ptensor1, ptensor0, swtensor1, swtensor0, fluidw)
        resvec2 = Resn(ptensor1, ptensor0, swtensor1, swtensor0, fluidn)
        resvec = torch.cat((resvec1, resvec2), dim=0)
        resnorm1 = torch.norm(resvec1, p=2).detach().cpu().numpy()
        resnorm2 = torch.norm(resvec1, p=2).detach().cpu().numpy()
        resnorm = torch.norm(resvec, p=2)
        loss = resnorm ** 2 / ncell / 2
        if (epoch + 1) % 100 == 0:
            print('epoch is ', epoch, 'resnorm is: ', resnorm1, ' and ', resnorm2)
            print('total is ', resnorm)
        if resnorm < tol:
            print('initial guess<tol=',tol)
            ptensor00 = ptensor1.clone().detach()
            swtensor00 = swtensor1.clone().detach()
            break
        elif resnorm<lowesttol:
            lowesttol=resnorm
            ptensor00 = ptensor1.clone().detach()
            swtensor00 = swtensor1.clone().detach()
        loss.backward()
        jac = predict.grad.clone()
        predict.grad.zero_()
        jacinv = jac / (jac.norm() ** 2)
        predict.data -= jacinv.data * loss
    print('final total epochs are ', epoch, 'resnorm is: ', resnorm1, ' and ', resnorm2)
    print('total is ', resnorm)
    return ptensor00, swtensor00, resnorm


def Jacobian2ph(pswvec1, pswvec0, fluidw, fluidn, dsp, dss):
    jac=torch.zeros((ncell*2,ncell*2), dtype=torch.float64)
    pvec1=pswvec1[:ncell]
    swvec1=pswvec1[ncell:]
    pvec0=pswvec0[:ncell]
    swvec0=pswvec0[ncell:]
    resvec_w1=Resw(pvec1, pvec0, swvec1, swvec0, fluidw)
    resvec_n1=Resn(pvec1, pvec0, swvec1, swvec0, fluidn)
    for j in range(ncell): # column-by-column for pressure
        pvec2=pvec1.detach().clone()
        pvec2[j]+= dsp
        resvec_w2 = Resw(pvec2, pvec0, swvec1, swvec0, fluidw)
        resvec_n2 = Resn(pvec2, pvec0, swvec1, swvec0, fluidn)
        jac[:ncell,j]=(resvec_w2-resvec_w1)/dsp
        jac[ncell:,j]=(resvec_n2-resvec_n1)/dsp
    for j in range(ncell): # column-by-column for sw
        swvec2 = swvec1.detach().clone()
        swvec2[j] += dss
        resvec_w2 = Resw(pvec1, pvec0, swvec2, swvec0, fluidw)
        resvec_n2 = Resn(pvec1, pvec0, swvec2, swvec0, fluidn)
        jac[:ncell, j+ncell] = (resvec_w2 - resvec_w1) / dss
        jac[ncell:, j+ncell] = (resvec_n2 - resvec_n1) / dss
    resvec=torch.concatenate((resvec_w1, resvec_n2),dim=0)
    return jac, resvec

def Newton2ph(pvec1, pvec0, swvec1, swvec0, fluidw, fluidn, dsp, dss, niter, tol):
    pswvec2 = torch.concatenate((pvec1, swvec1))
    pswvec0 = torch.concatenate((pvec0, swvec0))
    for i in range(niter):
        jac,resvec=Jacobian2ph(pswvec2,pswvec0,fluidw, fluidn, dsp, dss)
        resnorm=torch.linalg.norm(resvec)
        print('newton step ', i, ' residual: ',resnorm)
        if resnorm<tol:
            print('newton converge at step', i)
            break
        else:
            # invjac=np.linalg.inv(jac)
            # a=np.dot(invjac,resvec)
            # pswvec2 = pswvec2 - torch.matmul(torch.linalg.inv(jac),resvec)
            # ddx=torch.linalg.solve(jac, -1*resvec)
            ddx = scipy.sparse.linalg.spsolve(jac.detach().numpy(), -1 * resvec.detach().numpy(), permc_spec=None,use_umfpack=True)
            pswvec2 = pswvec2 + ddx
    if i==niter:
        jac,resvec=Jacobian2ph(pswvec2,pswvec0,fluidw, fluidn, dsp, dss)
        resnorm = torch.linalg.norm(resvec)
        print('newton step ', i, ' residual: ', resnorm)
        if resnorm>tol:
            print('newton doesnot converge at maxstep', i)
        else:
            print('newton converge at step', i)
    pvec2=pswvec2[:ncell]
    swvec2=pswvec2[ncell:]
    return pvec2, swvec2


# @CallingCounter

nx = 20; ny = 20; nz = 5; dx = 15.0; dy = 15.0; dz = 6.0
celllist,nodelist=BuildCartesianGrid(nx,ny,nz,dx,dy,dz)
ncell=len(celllist)
nnode=len(nodelist)
cellvolume_=dx*dy*dz
rock, fluidn, fluidw=SetRockFluid2ph()
PI,pwf, qwfix=SetWell()

p_init = 20.0 * 1e6
pvec_init, swvec_init=Setinitialcondition()
gvec=gravityvec()
pvec_init=torch.tensor(pvec_init,dtype=torch.float64)
swvec_init=torch.tensor(swvec_init,dtype=torch.float64)
gvec=torch.tensor(gvec,dtype=torch.float64)
nt = 60
dt = 200000
allh = np.loadtxt('perms3dme_realfirst.txt',skiprows=0)
chukvec=2.0 ** allh[0] * 0.1e-15
# chukvec=np.ones(ncell)*5e-15 # for homo
Buildtrans(chukvec)
neibortensor_0, neibortensor_1,neibortensor_2,neibortensor_3, neibortensor_4,neibortensor_5 = Buildneibortensor()
transvec_0, transvec_1, transvec_2, transvec_3, transvec_4, transvec_5 = Buildtranstensor()
zvec=Buildztensor()
markbcvec=Buildmarkbcvec()

starttime=time.time()
# allpress, allsw=twophase_compress(pvec_init, swvec_init, fluidw, fluidn, nt)
allpress=np.zeros((nt, ncell))
allsw=np.zeros((nt, ncell))


p_init = 20.0 * 1e6
pvec0=torch.ones(ncell,dtype=torch.float64)*p_init
swvec0=torch.ones(ncell,dtype=torch.float64)*0.2
tol1=1e-14
tol2 = 1e-14
predict = torch.zeros(ncell*2, dtype=torch.float64,requires_grad=True)
numofepochs=1000
for t in range(nt):
    print('Time is ', t)
    pvec1, swvec1, rr=ComputeInitialGuess(pvec0, swvec0, numofepochs, tol1)
    if rr>tol2:
        niter = 50
        dsp = 1e-3
        dss = 1e-6
        pressvec1, satwvec1 = Newton2ph(pvec1, pvec0, swvec1, swvec0, fluidw, fluidn, dsp, dss, niter, tol2)
        predict.data[:ncell] = (pressvec1.clone()-pvec0)/p_init
        predict.data[ncell:] = satwvec1.clone()-swvec0
        pvec0 = pressvec1.clone()
        swvec0 = satwvec1.clone()
    else:
        pvec0=pvec1
        swvec0=swvec1



allpress=pvec0.clone().detach().numpy()
allsw=swvec0.clone().detach().numpy()
endtime=time.time()
print(endtime-starttime)

# output
# qwt=np.zeros((4,nt))
# for ie in range(ncell):
#     cell_ = celllist[ie]
#     if cell_.markwell > 0:
#         qwt[cell_.markwell-1, 0] = qwt[cell_.markwell-1, 0]+PI * (allpress[0, ie] - p_init)
# for it in range(1, nt):
#     for ie in range(ncell):
#         cell_=celllist[ie]
#         if cell_.markwell>0:
#             qwt[cell_.markwell-1,it]=qwt[cell_.markwell-1,it]+PI*(allpress[it,ie]-allpress[it-1,ie])
# qwt=qwt.reshape(4*nt)
# np.savetxt('qwt-sample201.txt',np.abs(qwt))
# np.savetxt('allpress-steepinitial.txt',allpress)
# np.savetxt('allsw-steepinitial.txt',allsw)
# print("output to vtk")
# f = open('result_steepinitial_e-7.vtk','w')
# f.write("# vtk DataFile Version 2.0\n")
# f.write( "Unstructured Grid\n")
# f.write( "ASCII\n")
# f.write("DATASET UNSTRUCTURED_GRID\n")
# f.write("POINTS %d double\n" % (len(nodelist)))
# for i in range(0, len(nodelist)):
#     f.write("%0.3f %0.3f %0.3f\n" % (nodelist[i].x, nodelist[i].y, nodelist[i].z))
# f.write("\n")
# f.write("CELLS %d %d\n" % (len(celllist), len(celllist)*9))
# for i in range(0, len(celllist)):
#     f.write("%d %d %d %d %d %d %d %d %d\n" % (8, celllist[i].vertices[0], celllist[i].vertices[1], celllist[i].vertices[3], celllist[i].vertices[2], celllist[i].vertices[4], celllist[i].vertices[5], celllist[i].vertices[7], celllist[i].vertices[6]))
# f.write("\n")
# f.write("CELL_TYPES %d\n" % (len(celllist)))
# for i in range(0, len(celllist)):
#     f.write("12\n")
# f.write("\n")
# f.write("CELL_DATA %d\n" % (len(celllist)))
# f.write("SCALARS Permeability_mD double\n")
# f.write("LOOKUP_TABLE default\n")
# for i in range(0, len(celllist)):
#     f.write("%0.3f\n" % (chukvec[i]*1e15))
# f.write("SCALARS Pressure double\n")
# f.write("LOOKUP_TABLE default\n")
# for i in range(0, len(celllist)):
#     f.write("%0.3f\n" % (allpress[i]/10**6))
# f.write("SCALARS Sw double\n")
# f.write("LOOKUP_TABLE default\n")
# for i in range(0, len(celllist)):
#     f.write("%0.3f\n" % (allsw[i]))
# f.close()