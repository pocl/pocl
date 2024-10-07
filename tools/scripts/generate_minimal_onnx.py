"""
generate_minimal_onnx.py - construct a network that performs a logical XOR
operation with floats in the 0..1 range.

Copyright (c) 2024 Jan Solanti <jan.solanti@tuni.fi>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
"""

import numpy as np
import onnx
from onnx import numpy_helper
from onnx.helper import make_tensor_value_info, make_node, make_graph, make_model
from onnx.checker import check_model

A = make_tensor_value_info("A", onnx.TensorProto.FLOAT, ["length"])
B = make_tensor_value_info("B", onnx.TensorProto.FLOAT, ["length"])
C = make_tensor_value_info("C", onnx.TensorProto.FLOAT, ["length"])

IN_MIN = numpy_helper.from_array(np.array([0.0], dtype=np.float32), name="IN_MIN")
IN_MAX = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="IN_MAX")

node_clampA = make_node("Clip", ["A", "IN_MIN", "IN_MAX"], ["A_clamped"])
node_clampB = make_node("Clip", ["B", "IN_MIN", "IN_MAX"], ["B_clamped"])
node_add = make_node("Sub", ["A_clamped", "B_clamped"], ["sum1"])
node_abs = make_node("Abs", ["sum1"], ["abs1"])
node_threshold = make_node("Round", ["abs1"], ["C"])

graph = make_graph(
    [node_clampA, node_clampB, node_add, node_abs, node_threshold],
    "XOR",
    [A, B],
    [C],
    [IN_MIN, IN_MAX]
)
model = make_model(graph)
check_model(model)
print(model)
onnx.save(model, "xor_f32.onnx")
