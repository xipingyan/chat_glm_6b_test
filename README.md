# chat_glm_6b_test

Repos github: https://github.com/THUDM/ChatGLM-6B.git

```
pip install protobuf==3.20.0 transformers==4.27.1 icetk cpm_kernels
pip install onnxruntime
<!-- Run first(default pytorch) -->
python test.py
<!-- Convert to onnx model -->
python export_2_onnx.py
```

Convert ONNX ot OV IR
```
mo -m ./onnx_output_cg/chat_glm_6b.onnx -o ov_output --input "input_ids","position_ids","attention_mask","past_key_values" --output "lm_logits"
```