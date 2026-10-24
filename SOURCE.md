# Sources

This is an application extension using the official Atomman 1.5.4 source release from NIST-Software.

- Project: https://github.com/usnistgov/atomman
- Release: https://pypi.org/project/atomman/1.5.4/
- Source archive SHA256: `75b440bd88ed3fa1375fb4a5cfcf7143ddbc8d24f288889da9b3b690607bd811`.
- The bundled CPython 3.12 Linux wheel is built from that unmodified official archive. ElasticConstants, Stroh and SDVPN supply the native material, elastic tensor and semidiscrete energy conventions.
- Documentation: https://www.ctcms.nist.gov/potentials/atomman/atomman.defect.SDVPN.html

The GSFE tables and elastic parameters are synthetic. This extension uses a smooth periodic trigonometric interpolation of GammaSurface data and defines an effective overdamped stochastic model for selected dislocation-core migration branches. The resulting harmonic escape rates refer to those effective coordinates and mobility, rather than atomistic phonon frequencies.

Atomman is distributed under its included NIST terms. Licenses for the source and binary dependencies accompany the runtime artifacts.
