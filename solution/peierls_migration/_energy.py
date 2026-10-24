"""Periodic GSFE interpolation and semidiscrete dislocation energy."""
import numpy as np

import atomman as am

from atomman import unitconvert as uc

from scipy.optimize import root

TWOPI=2*np.pi

class Surface:
    def __init__(self, gamma):
        self.gamma=gamma
        values=gamma.data
        a=np.sort(values.a1.unique())
        b=np.sort(values.a2.unique())
        self.freq1=np.fft.fftfreq(len(a),1/len(a))
        self.freq2=np.fft.fftfreq(len(b),1/len(b))
        grid=values.pivot(index='a1',columns='a2',values='E_gsf').loc[a,b].values
        # A constant GSFE gauge contributes equally to all configurations.
        # Remove it before numerical energy differences are formed.
        self.coeff=np.fft.fft2(grid-grid.mean())/grid.size
        self.coeff[0,0]=0
        self.a=np.column_stack((gamma.a1vect,gamma.a2vect))

    def jet(self, z):
        e1=np.exp(1j*TWOPI*np.outer(z[:,0],self.freq1))
        e2=np.exp(1j*TWOPI*np.outer(z[:,1],self.freq2))
        c=self.coeff
        d1=1j*TWOPI*self.freq1[:,None]
        d2=1j*TWOPI*self.freq2[None,:]
        value=lambda c: np.einsum('ia,ab,ib->i',e1,c,e2).real
        v=value(c)
        g=np.column_stack((value(c*d1),value(c*d2)))
        h=np.empty((len(z),2,2))
        h[:,0,0]=value(c*d1*d1)
        h[:,0,1]=h[:,1,0]=value(c*d1*d2)
        h[:,1,1]=value(c*d2*d2)
        return v,g,h

def chi_matrix(n, spacing):
    d=np.subtract.outer(np.arange(n),np.arange(n))
    def psi(d):
        result=np.zeros(d.shape)
        nz=d!=0
        result[nz]=.5*(d[nz]*spacing)**2*np.log(abs(d[nz])*spacing)
        return result
    return 1.5*spacing**2+2*psi(d)-psi(d-1)-psi(d+1)

class Energy:
    def __init__(self, native, n, spacing, dissociated=False, partial='leading', width=5):
        self.native=native
        self.n=n
        self.surface=Surface(native.gamma)
        self.spacing=spacing
        self.dissociated=dissociated
        self.partial=partial
        self.width=width
        self.x=spacing*(np.arange(n+2)-(n+1)/2)
        difference=np.diff(np.eye(n+2),axis=0)/spacing
        chi=chi_matrix(n+1,spacing)
        a=self.surface.a
        k=a.T@native.K_tensor@a
        self.elastic=np.kron(difference.T@chi@difference,k)/(2*np.pi)
        self.force=np.zeros((n+2,2))
        stress=native.tau[1,:]@a
        weights=-.5*np.diff(self.x**2)/spacing
        self.force[:-1]-=weights[:,None]*stress
        self.force[1:]+=weights[:,None]*stress

    def full(self, z):
        result=np.zeros((self.n+2,2))
        result[1:-1]=np.asarray(z).reshape(self.n,2)
        result[-1,0]=1
        return result

    def jet(self,z,parameter=1):
        allz=self.full(z)
        vector=allz.ravel()
        v,g,h=self.surface.jet(allz)
        energy=self.spacing*sum(v)+self.force.ravel()@vector+.5*parameter*vector@self.elastic@vector
        gradient=self.spacing*g+self.force+(parameter*self.elastic@vector).reshape(self.n+2,2)
        hh=parameter*self.elastic.copy()
        ids=2*np.arange(self.n+2)
        hh[ids,ids]+=self.spacing*h[:,0,0]
        hh[ids+1,ids+1]+=self.spacing*h[:,1,1]
        hh[ids,ids+1]+=self.spacing*h[:,0,1]
        hh[ids+1,ids]+=self.spacing*h[:,1,0]
        return energy,gradient[1:-1].ravel(),hh[2:-2,2:-2]

    def initial(self,which):
        stress=self.native.tau[1,:]@self.surface.a
        loc=lambda z:self.surface.jet(np.asarray(z).reshape(1,2))[1][0]+stress
        low=root(loc,[0,0],tol=1e-12).x
        high=root(loc,[.5,0],tol=1e-12).x
        center=self.n//2
        z=np.tile(low,(self.n,1))
        if self.dissociated:
            left=center-self.width//2
            right=left+self.width-1
            middle=root(loc,[.5,0],tol=1e-12).x
            z[left:right+1]=middle
            z[right+1:,0]+=1
            if self.partial=='leading':
                center=right
                high=root(loc,[.75,0],tol=1e-12).x
                if which=='right': z[center]=low+[1,0]
            else:
                center=left
                high=root(loc,[.25,0],tol=1e-12).x
                z[center]=low
                if which=='right': z[center]=middle
        else:
            z[center+1:,0]+=1
            if which=='right': z[center,0]+=1
        if which=='saddle': z[center]=high
        return z

    def stationary(self,which):
        current=self.initial(which)
        parameter,step=0.,.03
        history=[]
        target_index=1 if which=='saddle' else 0
        while parameter<1:
            nxt=min(1,parameter+step)
            trial=root(lambda z:self.jet(z,nxt)[1],current.ravel(),jac=lambda z:self.jet(z,nxt)[2],tol=1e-11)
            energy,g,h=self.jet(trial.x,nxt)
            eig=np.linalg.eigvalsh(h)
            if max(abs(g))>1e-10 or np.count_nonzero(eig<0)!=target_index or max(abs(trial.x-current.ravel()))>.1:
                step*=.5
                if step<1e-9: raise ArithmeticError(f'{which} continuation failed at {parameter}, eig {eig[:3]}')
                continue
            current=trial.x.reshape(self.n,2)
            parameter=nxt
            history.append((parameter,eig[0],eig[1],max(abs(g))))
            step=min(.05,step*1.4)
        return current,np.asarray(history)
