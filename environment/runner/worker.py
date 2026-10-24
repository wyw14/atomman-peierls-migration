"""Native Atomman construction and protected solve with phase metadata."""
import contextlib
import importlib
import json
import os
import sys
import traceback

sys.path.insert(0, '/runner')
from codec import encode, encode_output, decode


def snapshot(model, settings):
    return json.dumps(encode({'K': model.K_tensor, 'burgers': model.burgers,
        'transform': model.transform, 'tau': model.tau, 'alpha': model.alpha,
        'beta': model.beta, 'data': model.gamma.data.to_dict('list'),
        'a1': model.gamma.a1vect, 'a2': model.gamma.a2vect,
        'settings': settings}), sort_keys=True, allow_nan=False)


def main():
    meta = {'wire_version': 1, 'worker_uid': os.geteuid()}
    stage = 'request_decode'
    try:
        request = decode(json.loads(sys.stdin.read()))
        with contextlib.redirect_stdout(sys.stderr):
            stage = 'native_construct'
            from material_factory import make_model
            material, settings = request['args']
            model = make_model(material)
            stage = 'input_snapshot'
            before = snapshot(model, settings)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        return {**meta, 'ok': False, 'stage': stage, 'input_measured': False,
                'error': type(error).__name__, 'message': str(error)}
    failure = None
    with contextlib.redirect_stdout(sys.stderr):
        stage = 'candidate_import'
        try:
            sys.path.insert(0, '/app/src')
            module = importlib.import_module('peierls_migration')
            stage = 'candidate_invoke'
            result = getattr(module, request['function'])(model, settings, **request['kwargs'])
        except Exception as error:
            failure = error
            traceback.print_exc(file=sys.stderr)
        finally:
            try:
                unchanged = before == snapshot(model, settings)
            except Exception:
                unchanged = False
    meta.update(stage=stage, input_measured=True, input_unchanged=unchanged)
    if failure is not None:
        return {**meta, 'ok': False, 'error': type(failure).__name__,
                'is_value_error': isinstance(failure, ValueError), 'message': str(failure)}
    try:
        answer = {**meta, 'ok': True, 'stage': 'completed', 'value': encode_output(result)}
        json.dumps(answer, allow_nan=False)
        return answer
    except Exception:
        return {**meta, 'ok': True, 'stage': 'completed',
                'value': {'__invalid_output__': 'unencodable result'}}


sys.stdout.write(json.dumps(main(), allow_nan=False))
