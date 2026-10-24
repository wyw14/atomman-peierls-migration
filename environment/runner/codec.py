"""Data-only numeric wire encoding; preserve malformed returned fields as data."""
import math
import numpy as np


def encode(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'__wire_float__': 'nan' if math.isnan(value) else '+inf' if value > 0 else '-inf'}
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'biufc':
            raise ValueError('Only numeric arrays are supported')
        return {'__wire_array__': {'shape': list(value.shape), 'data': encode(value.ravel().tolist())}}
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, complex):
        return {'__wire_complex__': [encode(value.real), encode(value.imag)]}
    if isinstance(value, tuple):
        return {'__wire_tuple__': [encode(x) for x in value]}
    if isinstance(value, list):
        return [encode(x) for x in value]
    if isinstance(value, dict):
        return {key: encode(x) for key, x in value.items()}
    return value


def encode_output(value, _depth=0):
    """A bad returned field is an ordinary mismatch, not a worker crash."""
    if _depth > 24:
        return {'__invalid_output__': 'nested value exceeds numeric output structure'}
    if isinstance(value, dict):
        result = {}
        for index, (key, item) in enumerate(value.items()):
            if not isinstance(key, str):
                key = '__invalid_output_key_' + str(index)
                while key in value or key in result:
                    key += '_'
            try:
                result[key] = encode_output(item, _depth + 1)
            except Exception:
                result[key] = {'__invalid_output__': 'unencodable field'}
        return result
    if isinstance(value, (list, tuple)):
        items = [encode_output(item, _depth + 1) for item in value]
        return {'__wire_tuple__': items} if isinstance(value, tuple) else items
    if isinstance(value, np.ndarray) and value.dtype.kind not in 'biufc':
        return {'__invalid_output__': 'nonnumeric array'}
    if isinstance(value, np.generic):
        return encode_output(value.item(), _depth + 1)
    if value is None or isinstance(value, (str, bool, int, float, complex, np.ndarray)):
        try:
            return encode(value)
        except Exception:
            return {'__invalid_output__': 'unencodable numeric value'}
    return {'__invalid_output__': 'unsupported value type'}


def decode(value):
    if isinstance(value, list):
        return [decode(x) for x in value]
    if isinstance(value, dict):
        if set(value) == {'__wire_float__'}:
            return {'nan': float('nan'), '+inf': float('inf'), '-inf': float('-inf')}[value['__wire_float__']]
        if set(value) == {'__wire_complex__'}:
            return complex(*(decode(x) for x in value['__wire_complex__']))
        if set(value) == {'__wire_tuple__'}:
            return tuple(decode(x) for x in value['__wire_tuple__'])
        if set(value) == {'__wire_array__'}:
            data = value['__wire_array__']
            return np.asarray(decode(data['data'])).reshape(tuple(data['shape']))
        return {key: decode(x) for key, x in value.items()}
    return value
