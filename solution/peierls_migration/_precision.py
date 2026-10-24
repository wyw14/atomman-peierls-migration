"""Adaptive precision for stationary states and soft collective curvature."""
import time
import mpmath as mp
import numpy as np


class _Problem:
    def __init__(self,model,settings,deadline):
        self.settings=settings
        self.deadline=deadline
        self.n=settings['n']
        self.a=mp.mpf(float(settings['spacing_A']))
        self.w=2*mp.pi
        table=model.gamma.data
        size=round(len(table)**.5)
        self.freq=range(-(size//2),size//2+1)
        values={(round(float(x)*size)%size,round(float(y)*size)%size):mp.mpf(float(v))
                for x,y,v in zip(table.a1,table.a2,table.E_gsf)}
        self.coeff=[]
        for k in self.freq:
            for l in self.freq:
                if k==l==0:continue
                c=sum(values[i,j]*mp.exp(-1j*self.w*(mp.mpf(k*i+l*j)/size))
                      for i in range(size) for j in range(size))/(size*size)
                self.coeff.append((k,l,c))
        basis=mp.matrix([[mp.mpf(float(model.gamma.a1vect[i])),mp.mpf(float(model.gamma.a2vect[i]))] for i in range(3)])
        tensor=mp.matrix([[mp.mpf(float(v)) for v in row] for row in model.K_tensor])
        tensor=(tensor+tensor.T)/2
        effective=basis.T*tensor*basis
        difference=mp.matrix(self.n+1,self.n+2)
        for i in range(self.n+1):difference[i,i]=-1/self.a;difference[i,i+1]=1/self.a
        psi=lambda d:mp.mpf(0) if d==0 else (d*self.a)**2*mp.log(abs(d)*self.a)/2
        chi=mp.matrix([[mp.mpf('1.5')*self.a*self.a+2*psi(i-j)-psi(i-j-1)-psi(i-j+1)
                       for j in range(self.n+1)] for i in range(self.n+1)])
        spatial=difference.T*chi*difference/(2*mp.pi)
        self.elastic=mp.matrix(2*(self.n+2),2*(self.n+2))
        for i in range(self.n+2):
            for j in range(self.n+2):
                for p in range(2):
                    for q in range(2):self.elastic[2*i+p,2*j+q]=spatial[i,j]*effective[p,q]
        traction=mp.matrix([[mp.mpf(float(v)) for v in model.tau[1,:]]])*basis
        self.force=mp.matrix(2*(self.n+2),1)
        x=[self.a*(mp.mpf(i)-mp.mpf(self.n+1)/2) for i in range(self.n+2)]
        for i in range(self.n+1):
            weight=-(x[i+1]**2-x[i]**2)/(2*self.a)
            for p in range(2):
                self.force[2*i+p]-=weight*traction[p]
                self.force[2*(i+1)+p]+=weight*traction[p]
        self.cache={}

    def check_time(self):
        if time.monotonic()>self.deadline:
            raise TimeoutError('Stationary-state precision budget exhausted')

    def jet(self,z):
        self.check_time()
        key=tuple(z)
        if key in self.cache:return self.cache[key]
        full=mp.matrix([0,0]+list(z)+[1,0])
        gradient=self.elastic*full+self.force
        hessian=self.elastic.copy()
        value=(full.T*self.elastic*full)[0]/2+(self.force.T*full)[0]
        for i in range(self.n+2):
            q,r=full[2*i],full[2*i+1]
            eq={k:mp.exp(1j*self.w*k*q) for k in self.freq}
            er={l:mp.exp(1j*self.w*l*r) for l in self.freq}
            v=gq=gr=hqq=hqr=hrr=mp.mpc(0)
            for k,l,c in self.coeff:
                term=c*eq[k]*er[l]
                v+=term;gq+=1j*self.w*k*term;gr+=1j*self.w*l*term
                hqq-=self.w*self.w*k*k*term
                hqr-=self.w*self.w*k*l*term
                hrr-=self.w*self.w*l*l*term
            value+=self.a*mp.re(v)
            gradient[2*i]+=self.a*mp.re(gq);gradient[2*i+1]+=self.a*mp.re(gr)
            hessian[2*i,2*i]+=self.a*mp.re(hqq)
            hessian[2*i+1,2*i+1]+=self.a*mp.re(hrr)
            hessian[2*i,2*i+1]+=self.a*mp.re(hqr)
            hessian[2*i+1,2*i]+=self.a*mp.re(hqr)
        result=value,gradient[2:-2,:],hessian[2:-2,2:-2]
        self.cache.clear();self.cache[key]=result
        return result


def _at_precision(model,settings,seeds,digits,deadline):
    with mp.workdps(digits):
        problem=_Problem(model,settings,deadline)
        states={}
        residual_limit=mp.power(10,-int(.65*digits))
        for name in ('left','right','saddle'):
            initial=[mp.mpf(float(v)) for v in np.asarray(seeds[name]).ravel()]
            state=mp.findroot(lambda *z:list(problem.jet(z)[1]),initial,
                J=lambda *z:problem.jet(z)[2],solver='mdnewton',tol=residual_limit,maxsteps=50)
            value,g,h=problem.jet(state)
            if max(abs(v) for v in g)>residual_limit:
                raise ArithmeticError('Unresolved critical branch at current precision')
            record={'state':state,'energy':value,'hessian':h,'residual':max(abs(v) for v in g)}
            if name!='saddle':
                chol=mp.cholesky(h)
                record['logdet']=2*sum(mp.log(chol[i,i]) for i in range(2*settings['n']))
            states[name]=record
            problem.check_time()
        saddle=states['saddle'];n=settings['n']
        active=n//2+(settings['plateau_width']//2 if settings['partial']=='leading' else -(settings['plateau_width']//2))
        curvature,mode=_eigenpair(saddle['hessian'],None,digits,problem)
        logdet_s=_one_negative_logdet(saddle['hessian'],mode,problem)
        if mode[2*active]<0:mode=-mode
        mobility=mp.matrix([[mp.mpf(float(v)) for v in row] for row in settings['mobility']])
        inverse=mobility**-1
        metric=mp.matrix(2*n,2*n)
        for i in range(n):
            for p in range(2):
                for q in range(2):metric[2*i+p,2*i+q]=inverse[p,q]
        drift_curvature,_=_eigenpair(saddle['hessian'],metric,digits,problem)
        growth=-drift_curvature
        if growth<=0:raise ArithmeticError('Unresolved unstable growth')
        barriers=[];logs=[]
        for name in ('left','right'):
            logdet=states[name]['logdet']
            barrier=saddle['energy']-states[name]['energy']
            barriers.append(barrier)
            logs.append(mp.log(mp.mpf(float(settings['frequency_s'])))+mp.log(growth)-mp.log(2*mp.pi)
                +(logdet-logdet_s)/2-mp.mpf(float(settings['line_length_A']))*barrier/(mp.mpf('8.617333262145e-5')*mp.mpf(float(settings['temperature_K']))))
        result={'minima':np.array([list(states[name]['state']) for name in ('left','right')],dtype=float).reshape(2,n,2),
            'saddle':np.array(list(saddle['state']),dtype=float).reshape(n,2),'barriers_eV_per_A':np.array(barriers,dtype=float),
            'negative_eigenvalue':float(curvature),'unstable_mode':np.array(list(mode),dtype=float).reshape(n,2),
            'growth_rate_s':float(mp.mpf(float(settings['frequency_s']))*growth),'log_rates_s':np.array(logs,dtype=float)}
        problem.check_time()
        return result,{'digits':digits,'maximum_stationary_residual':str(max(s['residual'] for s in states.values()))}


def _eigenpair(hessian,metric,digits,problem):
    """Correct one soft negative eigenpair of a symmetric definite pencil."""
    size=hessian.rows
    metric=mp.eye(size) if metric is None else metric
    hfloat=np.asarray(hessian.tolist(),dtype=float)
    gfloat=np.asarray(metric.tolist(),dtype=float)
    values,vectors=np.linalg.eig(np.linalg.solve(gfloat,hfloat))
    vector=mp.matrix([mp.mpf(float(v)) for v in vectors[:,np.argmin(values.real)].real])
    vector/=mp.sqrt((vector.T*vector)[0])
    value=(vector.T*hessian*vector)[0]/(vector.T*metric*vector)[0]
    absolute=mp.power(10,-int(.65*digits))
    relative=mp.power(10,-int(.35*digits))
    for _ in range(12):
        problem.check_time()
        gv=metric*vector
        residual=hessian*vector-value*gv
        norm=(vector.T*vector)[0]-1
        limit=min(absolute,abs(value)*relative)
        if max([abs(v) for v in residual]+[abs(norm)])<limit:
            if value>=0:raise ArithmeticError('Soft mode has the wrong inertia')
            vector/=mp.sqrt((vector.T*vector)[0])
            return (vector.T*hessian*vector)[0]/(vector.T*metric*vector)[0],vector
        jacobian=mp.matrix(size+1,size+1)
        for i in range(size):
            for j in range(size):jacobian[i,j]=hessian[i,j]-value*metric[i,j]
            jacobian[i,size]=-gv[i]
            jacobian[size,i]=2*vector[i]
        correction=mp.lu_solve(jacobian,mp.matrix(list(residual)+[norm]))
        vector-=correction[:size,:];value-=correction[size]
    raise ArithmeticError('Soft eigenpair correction did not converge')


def _one_negative_logdet(hessian,vector,problem):
    """Certify inertia using a positive orthogonal complement and a negative Schur scalar."""
    size=hessian.rows
    pivot=max(range(size),key=lambda i:abs(vector[i]))
    house=vector.copy();house[pivot]+=1 if vector[pivot]>=0 else -1
    denominator=(house.T*house)[0]
    columns=[j for j in range(size) if j!=pivot]
    q=mp.matrix(size,size-1)
    for i in range(size):
        for j,column in enumerate(columns):q[i,j]=(1 if i==column else 0)-2*house[i]*house[column]/denominator
    projected=q.T*hessian*q
    chol=mp.cholesky(projected)
    cross=q.T*hessian*vector
    first=mp.matrix(size-1,1)
    for i in range(size-1):first[i]=(cross[i]-sum(chol[i,j]*first[j] for j in range(i)))/chol[i,i]
    second=mp.matrix(size-1,1)
    for i in reversed(range(size-1)):second[i]=(first[i]-sum(chol[j,i]*second[j] for j in range(i+1,size-1)))/chol[i,i]
    schur=(vector.T*hessian*vector)[0]-(cross.T*second)[0]
    if schur>=0:raise ArithmeticError('Saddle inertia was not one')
    problem.check_time()
    return 2*sum(mp.log(chol[i,i]) for i in range(size-1))+mp.log(-schur)


def _converged(first,second):
    limits={'minima':(1e-9,1e-10),'saddle':(1e-9,1e-10),'barriers_eV_per_A':(1e-11,1e-9),
        'negative_eigenvalue':(1e-11,1e-9),'unstable_mode':(1e-9,1e-10),
        'growth_rate_s':(0,1e-9),'log_rates_s':(1e-8,0)}
    return all(np.allclose(first[key],second[key],atol=atol,rtol=rtol)
               for key,(atol,rtol) in limits.items())


def refine(model,settings,seeds,deadline):
    """Reconstruct the specified real energy until all returned fields stabilize."""
    previous=None;history=[]
    for digits in (45,75,115,170,250,370,550):
        if time.monotonic()>deadline:raise TimeoutError('Stationary-state precision budget exhausted')
        try:
            current,record=_at_precision(model,settings,seeds,digits,deadline)
        except (ValueError,ZeroDivisionError,ArithmeticError) as error:
            history.append({'digits':digits,'status':'unresolved','error':type(error).__name__})
            continue
        history.append(record)
        if previous is not None and _converged(previous,current):
            return current,history
        previous=current
        seeds={'left':current['minima'][0],'right':current['minima'][1],'saddle':current['saddle']}
    raise ArithmeticError('Critical-state precision did not stabilize')
