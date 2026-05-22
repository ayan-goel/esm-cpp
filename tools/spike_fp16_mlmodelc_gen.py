"""Generate one fp16 fc1 mlmodelc for the C++ BNNSGraph execute linchpin spike.
fc1 650M shape: input [M,K]=[2048,1280] fp32, weight [N,K]=[5120,1280] baked as
fp16 constant, output [M,N]=[2048,5120]. Compiled to .mlmodelc for
BNNSGraphCompileFromFile."""
import shutil, warnings, numpy as np
warnings.filterwarnings("ignore")
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

M, K, N = 2048, 1280, 5120
rng = np.random.default_rng(0)
W = (rng.standard_normal((N, K)).astype(np.float32) * 0.05)
B = (rng.standard_normal(N).astype(np.float32) * 0.01)

@mb.program(input_specs=[mb.TensorSpec(shape=(M, K), dtype=types.fp32)])
def prog(x):
    return mb.linear(x=x, weight=W, bias=B, name="out")

m = ct.convert(prog, convert_to="mlprogram",
               compute_units=ct.ComputeUnit.CPU_ONLY,
               compute_precision=ct.precision.FLOAT16,
               minimum_deployment_target=ct.target.macOS15)
cpath = m.get_compiled_model_path()  # temp .mlmodelc
dst = "/tmp/fc1_fp16.mlmodelc"
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(cpath, dst)
print("mlmodelc:", dst)
# Save reference input/output for the C++ correctness check.
X = (rng.standard_normal((M, K)).astype(np.float32) * 0.5)
Y = m.predict({"x": X})["out"].astype(np.float32)
X.tofile("/tmp/fc1_X.f32"); Y.tofile("/tmp/fc1_Y.f32")
print("ref io saved; Y[0,:3]=", Y[0, :3])
