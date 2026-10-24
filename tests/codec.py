"""Typed data transport with literal mappings and exact NumPy dtypes."""
import base64
import math
import numpy as np


def _pack_array(value):
    if value.dtype.kind not in 'biufcSU':
        raise ValueError('Unsupported array dtype')
    return {'dtype': value.dtype.str, 'shape': list(value.shape),
             'base64': base64.b64encode(value.tobytes(order='C')).decode('ascii')}


def _unpack_array(payload, numeric_only=False):
    if not isinstance(payload, dict) or set(payload) != {'dtype', 'shape', 'base64'}:
        raise ValueError('Invalid array envelope')
    if not isinstance(payload['dtype'], str):
        raise ValueError('Invalid array dtype')
    dtype = np.dtype(payload['dtype'])
    allowed = 'biufc' if numeric_only else 'biufcSU'
    if dtype.kind not in allowed or dtype.hasobject or dtype.itemsize <= 0:
        raise ValueError('Unsupported array dtype')
    shape = payload['shape']
    if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError('Invalid array shape')
    raw = base64.b64decode(payload['base64'], validate=True)
    if len(raw) != math.prod(shape) * dtype.itemsize:
        raise ValueError('Array byte count does not match shape and dtype')
    return np.frombuffer(raw, dtype=dtype).copy().reshape(tuple(shape))


def encode(value):
    if isinstance(value, np.ndarray):
        return {'__wire_array__': _pack_array(value)}
    if isinstance(value, np.generic):
        return {'__wire_scalar__': _pack_array(np.asarray(value))}
    if isinstance(value, float) and not math.isfinite(value):
        return {'__wire_float__': 'nan' if math.isnan(value) else '+inf' if value > 0 else '-inf'}
    if isinstance(value, complex):
        return {'__wire_complex__': [encode(value.real), encode(value.imag)]}
    if isinstance(value, tuple):
        return {'__wire_tuple__': [encode(x) for x in value]}
    if isinstance(value, list):
        return [encode(x) for x in value]
    if isinstance(value, dict):
        return {'__wire_dict__': [[encode(key), encode(item)] for key, item in value.items()]}
    return value


def encode_output(value, _depth=0):
    """A malformed field stays local data rather than failing the worker."""
    if _depth > 24:
        return {'__invalid_output__': 'nested value exceeds numeric output structure'}
    if isinstance(value, dict):
        entries = []
        used = set()
        for index, (key, item) in enumerate(value.items()):
            if not isinstance(key, str):
                key = '__invalid_output_key_' + str(index)
                while key in value or key in used:
                    key += '_'
            used.add(key)
            try:
                encoded = encode_output(item, _depth + 1)
            except Exception:
                encoded = {'__invalid_output__': 'unencodable field'}
            entries.append([encode(key), encoded])
        return {'__wire_dict__': entries}
    if isinstance(value, (list, tuple)):
        items = [encode_output(item, _depth + 1) for item in value]
        return {'__wire_tuple__': items} if isinstance(value, tuple) else items
    if isinstance(value, np.ndarray) and value.dtype.kind not in 'biufc':
        return {'__invalid_output__': 'nonnumeric array'}
    if isinstance(value, np.generic):
        try:
            return {'__wire_scalar__': _pack_array(np.asarray(value))}
        except Exception:
            return {'__invalid_output__': 'unsupported NumPy scalar'}
    if value is None or isinstance(value, (str, bool, int, float, complex, np.ndarray)):
        try:
            return encode(value)
        except Exception:
            return {'__invalid_output__': 'unencodable numeric value'}
    return {'__invalid_output__': 'unsupported value type'}


def _decode_data(value, _depth=0, *, numeric_only=False, decoder=None):
    if _depth > 24:
        return {'__invalid_output__': 'nested value exceeds numeric output structure'}
    recurse = decoder if decoder is not None else lambda item: _decode_data(
        item, _depth + 1, numeric_only=numeric_only)
    if isinstance(value, list):
        return [recurse(item) for item in value]
    if not isinstance(value, dict):
        return value
    try:
        if set(value) == {'__wire_dict__'}:
            entries = value['__wire_dict__']
            if not isinstance(entries, list) or any(
                    not isinstance(pair, list) or len(pair) != 2 for pair in entries):
                raise ValueError('Invalid mapping envelope')
            # Return the reconstructed mapping directly. Its literal keys are
            # user data and must never be interpreted again as wire tags.
            return {recurse(key): recurse(item) for key, item in entries}
        if set(value) == {'__wire_float__'}:
            return {'nan': float('nan'), '+inf': float('inf'), '-inf': float('-inf')}[value['__wire_float__']]
        if set(value) == {'__wire_complex__'}:
            return complex(*(recurse(item) for item in value['__wire_complex__']))
        if set(value) == {'__wire_tuple__'}:
            return tuple(recurse(item) for item in value['__wire_tuple__'])
        if set(value) == {'__wire_array__'}:
            return _unpack_array(value['__wire_array__'], numeric_only=numeric_only)
        if set(value) == {'__wire_scalar__'}:
            scalar = _unpack_array(value['__wire_scalar__'], numeric_only=numeric_only)
            if scalar.shape != ():
                raise ValueError('NumPy scalar must have zero dimensions')
            return scalar[()]
        if set(value) == {'__wire_ngsolve_mesh__'}:
            return {'__invalid_output__': 'native mesh is not a numeric output',
                     'raw_native_wire': value['__wire_ngsolve_mesh__']}
    except Exception:
        return {'__invalid_output__': 'malformed numeric wire value', 'raw': value}
    return {key: recurse(item) for key, item in value.items()}


def decode(value):
    return _decode_data(value)


