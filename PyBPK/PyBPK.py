#importing dependencies
import datetime
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

#class for tissues like kidney, spleen, liver, kleenex, etc...
class Tissue:
    _allTissues = []
    _tissues = []
    _plasma = []
    def __init__(self,name,volume,flow,distRatio,conc=0,klear=0,isPlasma=False,outlet=['plasma']):
        self.isPlasma = isPlasma
        self.v = volume
        self.c = conc
        Tissue._allTissues.append(self)
        if isPlasma:
            self.n = 'plasma'
            self.q = 0
            self.r = 1
            self.k = 0
            self.out = Tissue._tissues
            Tissue._plasma.append(self)
        else:
            self.n = name
            self.q = flow
            self.r = distRatio
            self.k = klear
            self.out = outlet
            Tissue._tissues.append(self)
        if len(Tissue._plasma) > 1:
            print("You have more than one tissue designated as plasma! This will cause issues")
    def deriv(self,plasma):
        Qall = []
        Q = [self.q]
        C = [plasma.c]
        R = [1]
        for tissue in Tissue._tissues:
            Qall.append(tissue.q)
            if self.n in tissue.out:
                Q.append(tissue.q)
                C.append(tissue.c)
                R.append(tissue.r)
        Qall = np.array(Qall)
        Q = np.array(Q)
        C = np.array(C)
        R = np.array(R)
        if not self.isPlasma:
            return (1/self.v)*(np.dot(np.true_divide(Q,R),C)-(self.c/self.r)*(self.q+self.k))
        else:
            return (1/self.v)*(np.dot(np.true_divide(Q,R),C)-self.c*sum(Qall))
       
#pharmacokinetic model object
class Model():
    def __init__(self,dose,simTime,plot,timedDose,doseInterval,saveOutput):
        self.tf = simTime
        self.ti = (0,simTime)
        self.len = self.ti[1]-self.ti[0]
        self.dose = dose
        self.di = doseInterval
        x0 = np.zeros(len(Tissue._allTissues))
        x0[0] = (1/Tissue._plasma[0].v)*self.dose[0]
        if not timedDose:
            sol = solve_ivp(dxdt,self.ti,x0)
            self.t = sol.t
            self.x = sol.y
        if timedDose:
            iterations = math.ceil(self.len/self.di)
            sol = solve_ivp(dxdt,(0,self.di),x0)
            x = sol.y
            x[0,-1] += (1/Tissue._plasma[0].v)*self.dose[1]
            self.x = x
            t = sol.t
            self.t = t
            for i in range(iterations-1):
                sol = solve_ivp(dxdt,(t[-1],min(t[-1]+self.di,self.ti[1])),x[:,-1])
                x = sol.y
                if i != (iterations-2):
                    if i <= (len(dose)-3):
                        x[0,-1] += (1/Tissue._plasma[0].v)*self.dose[i+2]
                    else:
                        x[0,-1] += (1/Tissue._plasma[0].v)*self.dose[-1]
                t = sol.t
                self.x = np.append(self.x[:,:-1],x,axis=1)
                self.t = np.append(self.t,t[1:],axis=0)      
        if plot:
            for i in range(len(Tissue._allTissues)):
                plt.plot(self.t,self.x[i,:],label=Tissue._allTissues[i].n)
            plt.legend()
        if saveOutput:
            d = datetime.datetime.now()
            self.data = np.ma.row_stack((self.t,self.x))
            np.savetxt("{:%m%d%y%H%M%S}".format(d)+".csv",self.data,delimiter=",")
            plt.savefig("{:%m%d%y%H%M%S}".format(d)+".png",dpi=500)

def clearTissues():
    Tissue._allTissues = []
    Tissue._tissues = []
    Tissue._plasma = []
    print('Tissues Cleared')

#find entry in numpy array closest to a given value
def findNearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
                  
#function for making plasma, wraps around unnecessary tissue attributes
def makePlasma(volume,concentration=0):
    return Tissue('plasma',volume,0,1,conc=concentration,isPlasma=True)

#function that wraps around Model class so doses may be given as lists or as floats
def makeModel(dose,simTime,plot=True,timedDose=False,doseInterval=0,saveOutput=True):
    if not isinstance(dose, list):
        dose = [dose]
    return Model(dose,simTime,plot,timedDose,doseInterval,saveOutput)
        
#derivative function wrapped around tissue.deriv() to work with scipy  
def dxdt(t,x):
    dxdt = np.zeros(len(Tissue._allTissues))
    for i in range(len(Tissue._allTissues)):
        Tissue._allTissues[i].c = x[i]
    for i in range(len(Tissue._allTissues)):
        dxdt[i] = Tissue._allTissues[i].deriv(Tissue._plasma[0])
    return dxdt

def cost(R,toOptimize,data,dose,timed,interval): #data must have time as first row, and then concentrations for each training organ as a row vector in order
    for i in range(len(toOptimize)):
        toOptimize[i].r = R[i]
    mod = Model(dose,data[0,-1],plot=False,timedDose=timed,doseInterval=interval)
    t = data[0,:]
    x = data[1:,:]
    that = mod.t[findNearest(mod.t,t[0])]
    xhat = mod.x[:,findNearest(mod.t,t[0])]
    for i in range(len(t)-1):
        index = findNearest(mod.t,t[i+1])
        that = np.append(that, mod.t[index])
        xhat = np.column_stack((xhat, mod.x[:,index]))
    datahat = np.ma.row_stack((that,xhat))
    err = []
    for i in range(len(t)): #Euclidean distance from actual state, will be minimized
        v = x[:,i]
        vhat = datahat[1:,i]
        dist = np.linalg.norm(v-vhat)
        err.append(dist)
    return err

def optimize(toOptimize,data,dose,timed=False,interval=1):
    R0 = np.ones(len(toOptimize),)
    optimizeResult = least_squares(cost,R0,bounds=(1,np.inf),args=(toOptimize,data,dose,timed,interval))
    for i in range(len(toOptimize)):
        toOptimize[i].r = optimizeResult.x[i]