"""Independent energy-path certificates for dislocation migration."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from candidate import call

CASES = json.loads((Path(__file__).parent/'fixtures.json').read_text())
FIELDS = ['minima', 'saddle', 'barriers_eV_per_A', 'negative_eigenvalue',
          'unstable_mode', 'growth_rate_s', 'log_rates_s']
KEYS = set(FIELDS)
TOLERANCES = {
    'minima': (2e-6, 0), 'saddle': (2e-6, 0),
    'barriers_eV_per_A': (1e-7, 2e-6),
    'negative_eigenvalue': (1e-7, 8e-6),
    'unstable_mode': (3e-6, 0), 'growth_rate_s': (0, 8e-6),
    'log_rates_s': (2e-4, 0)}


class Deferred:
    """Cache a raw invocation, keeping independent field assertions separate."""
    def __init__(self, material, settings):
        self.args = (material, settings)
        self.loaded, self.result, self.error = False, None, None

    def load(self):
        if not self.loaded:
            self.loaded = True
            try:
                self.result = call('solve', *self.args)
            except Exception as error:
                self.error = error

    def get(self):
        self.load()
        if self.error is not None:
            raise self.error
        return self.result


def field_value(actual, expected, field):
    assert isinstance(actual, dict), 'output must be a dictionary'
    assert field in actual, 'missing field: '+field
    value = np.asarray(actual[field])
    assert value.shape == np.asarray(expected[field]).shape, (field, value.shape)
    assert value.dtype.kind in 'fiu', (field, value.dtype)
    assert np.isfinite(value).all(), 'nonfinite field: '+field
    return value


def complete_schema(actual, settings):
    n = settings['n']
    shapes = {'minima': (2,n,2), 'saddle': (n,2), 'barriers_eV_per_A': (2,),
        'negative_eigenvalue': (), 'unstable_mode': (n,2),
        'growth_rate_s': (), 'log_rates_s': (2,)}
    assert isinstance(actual, dict) and set(actual) == KEYS, 'output keys'
    for field, shape in shapes.items():
        value = np.asarray(actual[field])
        assert value.shape == shape, (field, value.shape, shape)
        assert value.dtype.kind in 'fiu' and np.isfinite(value).all(), field


def compare_field(actual, expected, field):
    value = field_value(actual, expected, field)
    atol, rtol = TOLERANCES[field]
    np.testing.assert_allclose(value, expected[field], atol=atol, rtol=rtol, err_msg=field)


@pytest.fixture(scope='session', params=CASES, ids=lambda c:c['id'])
def computed(request):
    case = request.param
    return case, Deferred(case['material'], case['settings'])


def test_schema(computed):
    case, actual = computed
    complete_schema(actual.get(), case['settings'])


def test_minima(computed):
    case, actual = computed
    compare_field(actual.get(), case['expected'], 'minima')


def test_saddle(computed):
    case, actual = computed
    compare_field(actual.get(), case['expected'], 'saddle')


def test_barriers(computed):
    case, actual = computed
    compare_field(actual.get(), case['expected'], 'barriers_eV_per_A')


def test_negative_curvature(computed):
    case, actual = computed
    compare_field(actual.get(), case['expected'], 'negative_eigenvalue')


def test_unstable_mode(computed):
    case, actual = computed
    compare_field(actual.get(), case['expected'], 'unstable_mode')


def test_mobility_growth(computed):
    case, actual = computed
    compare_field(actual.get(), case['expected'], 'growth_rate_s')


def test_harmonic_rates(computed):
    case, actual = computed
    compare_field(actual.get(), case['expected'], 'log_rates_s')


BAD_SETTINGS = [('n',12),('n',True),('plateau_width',4),('plateau_width',11),
    ('partial','other'),('temperature_K',0),('frequency_s',-1),
    ('mobility',[[1,.2],[.3,1]]),('mobility',[[1,2],[2,1]])]


@pytest.mark.parametrize('field,value', BAD_SETTINGS)
def test_invalid_settings(field, value):
    case = CASES[0]
    settings = copy.deepcopy(case['settings'])
    settings[field] = value
    with pytest.raises(ValueError):
        call('solve', case['material'], settings)


def test_missing_setting():
    case = CASES[0]
    settings = copy.deepcopy(case['settings'])
    del settings['line_length_A']
    with pytest.raises(ValueError):
        call('solve', case['material'], settings)


@pytest.fixture(scope='module')
def repeated():
    case = CASES[3]
    results = [Deferred(case['material'], case['settings']) for _ in range(2)]
    # Both actual calls occur before a per-field ordinary comparison can fail.
    for result in results:
        result.load()
    return case, results


@pytest.mark.parametrize('index', [0,1])
def test_repeated_call_schema(repeated, index):
    case, results = repeated
    complete_schema(results[index].get(), case['settings'])


@pytest.mark.parametrize('field', FIELDS)
def test_repeated_call_field(repeated, field):
    case, results = repeated
    first, second = (result.get() for result in results)
    compare_field(first, case['expected'], field)
    compare_field(second, case['expected'], field)
    atol, rtol = TOLERANCES[field]
    np.testing.assert_allclose(field_value(second, case['expected'], field),
        field_value(first, case['expected'], field), atol=atol, rtol=rtol, err_msg=field)


TRANSFORMS = ['line-length', 'mobility-rescaling', 'surface-energy-gauge']


def transformed_case(name):
    index = {'line-length':0, 'mobility-rescaling':1, 'surface-energy-gauge':4}[name]
    case = copy.deepcopy(CASES[index])
    expected = {field:np.asarray(value).copy() for field,value in case['expected'].items()}
    if name == 'line-length':
        original_length = case['settings']['line_length_A']
        case['settings']['line_length_A'] *= 2
        expected['log_rates_s'] -= original_length*expected['barriers_eV_per_A']/(8.617333262145e-5*case['settings']['temperature_K'])
    elif name == 'mobility-rescaling':
        case['settings']['mobility'] = (2*np.asarray(case['settings']['mobility'])).tolist()
        expected['growth_rate_s'] *= 2
        expected['log_rates_s'] += np.log(2)
    elif name == 'surface-energy-gauge':
        case['material']['offset_eV_A2'] += .081
    return case, expected


@pytest.fixture(scope='module', params=TRANSFORMS)
def transformed(request):
    case, expected = transformed_case(request.param)
    return case, expected, Deferred(case['material'], case['settings'])


def test_transformed_schema(transformed):
    case, expected, actual = transformed
    complete_schema(actual.get(), case['settings'])


@pytest.mark.parametrize('field', FIELDS)
def test_transformed_physical_outputs(transformed, field):
    case, expected, actual = transformed
    compare_field(actual.get(), expected, field)
