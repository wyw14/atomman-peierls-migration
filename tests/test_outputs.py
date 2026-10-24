"""Independent energy-path certificates for dislocation migration."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from candidate import call

CASES=json.loads((Path(__file__).parent/'fixtures.json').read_text())
KEYS={'minima','saddle','barriers_eV_per_A','negative_eigenvalue',
      'unstable_mode','growth_rate_s','log_rates_s'}


@pytest.fixture(scope='session',params=CASES,ids=lambda c:c['id'])
def computed(request):
    case=request.param
    actual=call('solve',case['material'],case['settings'])
    return case,actual


def test_schema(computed):
    """All seven physical outputs have finite real values and declared shapes."""
    case,actual=computed
    n=case['settings']['n']
    shapes={'minima':(2,n,2),'saddle':(n,2),'barriers_eV_per_A':(2,),
        'negative_eigenvalue':(),'unstable_mode':(n,2),'growth_rate_s':(),'log_rates_s':(2,)}
    assert isinstance(actual,dict)
    assert set(actual)==KEYS
    for key,shape in shapes.items():
        value=np.asarray(actual[key])
        assert value.shape==shape
        assert value.dtype.kind in 'fiu'
        assert np.isfinite(value).all()


def test_minima(computed):
    """Both selected metastable cores agree with independent energy minimization."""
    case,actual=computed
    np.testing.assert_allclose(actual['minima'],case['expected']['minima'],atol=2e-6,rtol=0)


def test_saddle(computed):
    """The index-one core agrees with a constrained minimum-energy path maximum."""
    case,actual=computed
    np.testing.assert_allclose(actual['saddle'],case['expected']['saddle'],atol=2e-6,rtol=0)


def test_barriers(computed):
    """Forward and reverse eV/angstrom barriers use nonlocal anisotropic core energies."""
    case,actual=computed
    np.testing.assert_allclose(actual['barriers_eV_per_A'],case['expected']['barriers_eV_per_A'],atol=1e-7,rtol=2e-6)


def test_negative_curvature(computed):
    """The full free-coordinate Hessian has the independently evaluated negative curvature."""
    case,actual=computed
    np.testing.assert_allclose(actual['negative_eigenvalue'],case['expected']['negative_eigenvalue'],atol=1e-7,rtol=8e-6)


def test_unstable_mode(computed):
    """The collective unstable vector retains transverse relaxation and the stated sign."""
    case,actual=computed
    np.testing.assert_allclose(actual['unstable_mode'],case['expected']['unstable_mode'],atol=3e-6,rtol=0)


def test_mobility_growth(computed):
    """Growth follows the mobility-weighted Hessian, independently symmetrized."""
    case,actual=computed
    np.testing.assert_allclose(actual['growth_rate_s'],case['expected']['growth_rate_s'],atol=0,rtol=8e-6)


def test_harmonic_rates(computed):
    """Log rates include full-coordinate fluctuations and line-length thermal energy."""
    case,actual=computed
    np.testing.assert_allclose(actual['log_rates_s'],case['expected']['log_rates_s'],atol=2e-4,rtol=0)


BAD_SETTINGS=[('n',12),('n',True),('plateau_width',4),('plateau_width',11),
    ('partial','other'),('temperature_K',0),('frequency_s',-1),
    ('mobility',[[1,.2],[.3,1]]),('mobility',[[1,2],[2,1]])]


@pytest.mark.parametrize('field,value',BAD_SETTINGS)
def test_invalid_settings(field,value):
    """Unsupported settings raise ValueError without changing caller inputs."""
    case=CASES[0]
    settings=copy.deepcopy(case['settings']);settings[field]=value
    with pytest.raises(ValueError):
        call('solve',case['material'],settings)


def test_missing_setting():
    """A missing required setting raises the documented exception type."""
    case=CASES[0]
    settings=copy.deepcopy(case['settings']);del settings['line_length_A']
    with pytest.raises(ValueError):
        call('solve',case['material'],settings)


def test_repeated_call():
    """Repeated native-object calls agree and preserve their inputs."""
    case=CASES[3]
    first=call('solve',case['material'],case['settings'])
    second=call('solve',case['material'],case['settings'])
    for key in KEYS:
        np.testing.assert_allclose(first[key],second[key],atol=2e-10,rtol=2e-12)


def test_line_length_thermal_scaling():
    """Line length converts the per-length barrier to the Boltzmann energy."""
    case=CASES[0]
    settings=copy.deepcopy(case['settings']);settings['line_length_A']*=2
    result=call('solve',case['material'],settings)
    expected=np.asarray(case['expected']['log_rates_s'])-case['settings']['line_length_A']*np.asarray(case['expected']['barriers_eV_per_A'])/(8.617333262145e-5*case['settings']['temperature_K'])
    np.testing.assert_allclose(result['log_rates_s'],expected,atol=2e-4,rtol=0)


def test_mobility_rescaling():
    """A uniform mobility change rescales drift rates without changing the energy landscape."""
    case=CASES[1]
    settings=copy.deepcopy(case['settings']);settings['mobility']=(2*np.asarray(settings['mobility'])).tolist()
    result=call('solve',case['material'],settings)
    np.testing.assert_allclose(result['minima'],case['expected']['minima'],atol=2e-6,rtol=0)
    np.testing.assert_allclose(result['saddle'],case['expected']['saddle'],atol=2e-6,rtol=0)
    np.testing.assert_allclose(result['growth_rate_s'],2*case['expected']['growth_rate_s'],atol=0,rtol=8e-6)
    np.testing.assert_allclose(result['log_rates_s'],np.asarray(case['expected']['log_rates_s'])+np.log(2),atol=2e-4,rtol=0)


def test_surface_energy_gauge():
    """A constant GSFE offset leaves migration states, barriers and rates unchanged."""
    case=CASES[4]
    material=copy.deepcopy(case['material']);material['offset_eV_A2']+=.081
    result=call('solve',material,case['settings'])
    for key in ('minima','saddle','barriers_eV_per_A','unstable_mode'):
        np.testing.assert_allclose(result[key],case['expected'][key],atol=3e-6,rtol=2e-6)
    np.testing.assert_allclose(result['log_rates_s'],case['expected']['log_rates_s'],atol=2e-4,rtol=0)
