# Numerical reference for CVIO core

`cvio_core_golden.json` was computed directly from the unmodified upstream
`losses/asl_ldam.py` and `attention/simam_dcfr.py` at
`fd64876cd2fd3acb24640b729d4f019996ebb39b`, using PyTorch 2.9.1 CPU, float32.
Source file SHA-256 values are embedded in the fixture. Source license: AGPL-3.0;
see `training/cvio_core/NOTICE` and `LICENSE`.

Loss inputs/targets use all four classes and the upstream defaults. For attention,
the input is linspace(-0.9, 1.1, 48), shape (1, 3, 4, 4). Each parameter number i
uses linspace(-0.15 + i*0.02, 0.15 + i*0.02, numel), in upstream declaration order.
The backward objective is output.square().mean(). All resulting weights, output,
input gradients and parameter gradients are stored. Regression tolerance is
rtol=atol=1e-6. Do not regenerate expected values from the port under test.
