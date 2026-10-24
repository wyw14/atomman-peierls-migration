"""Native Atomman material construction with data-only candidate transport."""
import contextlib
import importlib
import json
import sys
import traceback

sys.path.insert(0,'/runner')
from codec import encode,decode

request=decode(json.loads(sys.stdin.read()))
try:
    with contextlib.redirect_stdout(sys.stderr):
        from material_factory import make_model
        material,settings=request['args']
        model=make_model(material)
        def snapshot():
            return json.dumps(encode({'K':model.K_tensor,'burgers':model.burgers,
                'transform':model.transform,'tau':model.tau,'alpha':model.alpha,
                'beta':model.beta,'data':model.gamma.data.to_dict('list'),
                'a1':model.gamma.a1vect,'a2':model.gamma.a2vect,'settings':settings}),
                sort_keys=True,allow_nan=False)
        before=snapshot()
        sys.path.insert(0,'/app/src')
        module=importlib.import_module('peierls_migration')
        failure=None
        try:
            result=getattr(module,request['function'])(model,settings,**request['kwargs'])
        except Exception as error:
            failure=error
        if snapshot()!=before:
            raise AssertionError('Caller input was modified')
        if failure is not None:
            raise failure
    answer={'ok':True,'value':encode(result)}
except Exception as error:
    traceback.print_exc(file=sys.stderr)
    answer={'ok':False,'error':type(error).__name__,'message':str(error)}
sys.stdout.write(json.dumps(answer,allow_nan=False))
