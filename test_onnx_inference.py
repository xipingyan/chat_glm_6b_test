
import onnxruntime as ort
import numpy as np
import onnx

onnx_model = 'onnx_output_auto/chat_glm_6b.onnx'
# Load the ONNX model
model = onnx.load(onnx_model)
onnx.checker.check_model(model)
# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

# ort_sess = ort.InferenceSession()
# test = "你好"
# test = "晚上睡不着应该怎么办"

# outputs = ort_sess.run(None, {'input': text.numpy(),
#                             'offsets':  torch.tensor([0]).numpy()})
# # Print Result
# result = outputs[0].argmax(axis=1)+1
# print("This is a %s news" %ag_news_label[result[0]])
