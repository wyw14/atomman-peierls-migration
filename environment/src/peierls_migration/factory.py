"""Construct native Atomman objects from a synthetic GSFE material model."""
import atomman as am
from atomman import unitconvert as uc
import numpy as np


def make_model(material):
    """Build the supported SDVPN material in the dislocation Cartesian frame."""
    elastic=am.ElasticConstants(Cij=uc.set_in_units(np.asarray(material['cij_gpa']),'GPa'))
    if 'rotation' in material:
        elastic=elastic.transform(np.asarray(material['rotation']))
    b=float(material.get('burgers_A',2.5))
    a1=np.array([b,0,0])
    a2=np.array([float(material['skew'])*b,0,float(material['transverse_ratio'])*b])
    volterra=am.defect.Stroh(elastic,burgers=a1)
    grid=int(material.get('grid',9))
    points=np.arange(grid)/grid
    q,r=np.meshgrid(points,points,indexing='ij')
    w=2*np.pi
    amplitude=float(material['amplitude_eV_A2'])
    fault=float(material['fault_ratio'])
    transverse=float(material['transverse_ratio_energy'])
    cross=float(material['cross_ratio'])
    value=amplitude*(1-np.cos(2*w*q)+fault*(1-np.cos(w*q))+transverse*(1-np.cos(w*r))+cross*np.sin(2*w*q)*np.sin(w*r))
    value+=float(material.get('offset_eV_A2',0))
    gamma=am.defect.GammaSurface(a1vect=a1,a2vect=a2,a1=q.ravel(),a2=r.ravel(),E_gsf=value.ravel())
    stress=np.zeros((3,3))
    traction=uc.set_in_units(np.asarray(material['traction_gpa']),'GPa')
    stress[1,:]=traction;stress[:,1]=traction
    return am.defect.SDVPN(volterra=volterra,gamma=gamma,tau=stress,
                          alpha=[0.0],beta=np.zeros((3,3)),fullstress=True,
                          cdiffelastic=False,cdiffstress=False)
