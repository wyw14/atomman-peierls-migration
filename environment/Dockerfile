FROM python:3.12.15-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f
ENV PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
COPY wheels/ /tmp/wheels/
COPY requirements.runtime.lock.txt /tmp/requirements.runtime.lock.txt
RUN python -m pip install --no-cache-dir --require-hashes --only-binary=:all: --no-deps -r /tmp/requirements.runtime.lock.txt \
    && python -m pip check \
    && python -c "import mpmath,gmpy2; assert mpmath.__version__=='1.3.0' and gmpy2.version()=='2.2.1' and mpmath.libmp.BACKEND=='gmpy'; import atomman as am,numpy as np; from atomman import unitconvert as uc; assert am.__version__=='1.5.4'; t=uc.get_in_units(am.ElasticConstants(Cij=uc.set_in_units(np.diag([2e11,2e11,2e11,8e10,8e10,8e10]),'Pa')).Cijkl,'Pa'); assert t.shape==(3,3,3,3) and np.isclose(t[0,0,0,0],2e11) and np.isclose(t[0,1,0,1],8e10)" \
    && rm -rf /tmp/wheels /tmp/requirements.runtime.lock.txt

WORKDIR /app
COPY src/ /app/src/
COPY runner/ /runner/
ENV PYTHONPATH=/app/src MPLBACKEND=Agg
