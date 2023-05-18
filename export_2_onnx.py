import os
import torch
from transformers import AutoTokenizer, AutoModel

model_path="./local_chat_glm_6b_model"

#Auto
def ConvertAutoModel(output_dir, output_onnx):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    onnx_path = output_dir + "/" + output_onnx

    print("--> Load tokenizer and model.")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to("cpu", dtype=float)

    import numpy as np
    input_ids = torch.tensor(np.array([[5, 74874, 130001, 130004]]), dtype=torch.long)

    # switch model to inference mode
    print("--> model.transformer.eval.")
    model.eval()

    # disable gradients calculation for reducing memory consumption
    print("--> export.")
    with torch.no_grad():
        # export model to ONNX format
        torch.onnx.export(
            model,  # model instance
            input_ids,  # inputs for model tracing
            f=onnx_path,  # output file for saving result
            input_names=["input_ids"],  # model input name for onnx representation
            output_names=["lm_logits"],
        )
    print('--> ChatGLM-6B successfully converted to ONNX.')

# ConditionalGeneration Model
from local_chat_glm_6b_model.tokenization_chatglm import ChatGLMTokenizer
from local_chat_glm_6b_model.modeling_chatglm import ChatGLMForConditionalGeneration
def ConvertCGModel(output_dir, output_onnx):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    onnx_path = output_dir + "/" + output_onnx

    tokenizer = ChatGLMTokenizer.from_pretrained(model_path) 
    model = ChatGLMForConditionalGeneration.from_pretrained(model_path).float().cpu()
    model = model.eval() 
    device="cpu" 

    input_text = "你好" 
    input_ids = tokenizer([input_text], return_tensors="pt")["input_ids"]
    # print("input_ids =", input_ids) 
    position_ids = torch.tensor([[[0, 1, 2, 2], [0, 0, 0, 1]]]) 
    attention_mask = torch.tensor(
        [[[
            [False, False, False, True],
            [False, False, False, True],
            [False, False, False, True],
            [False, False, False, False]     ]]] ) 
    past_key_values = torch.zeros(28, 2, 0, 1, 32, 128) 
    dynamic_axes={        
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 2: "seq_len"},
            "attention_mask": {0: "batch_size", 2: "seq_len", 3: "seq_len"},
            "past_key_values": {2: "history_len"} } 
    torch.onnx.export(
        model,
        f = onnx_path,
        args=(input_ids, position_ids, attention_mask, past_key_values),
        input_names=["input_ids", "position_ids", "attention_mask", "past_key_values"],
        output_names=["lm_logits"],
        dynamic_axes=dynamic_axes,
        opset_version=14, )

def ConvertCGModel_myupdate(output_dir, output_onnx):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    onnx_path = output_dir + "/" + output_onnx

    tokenizer = ChatGLMTokenizer.from_pretrained(model_path) 
    # model = ChatGLMForConditionalGeneration.from_pretrained(model_path).float().cpu()
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to("cpu", dtype=float)
    print(model.transformer)

    # model = model.eval() 
    # device="cpu" 

    # input_text = "你好" 
    # input_ids = tokenizer([input_text], return_tensors="pt")["input_ids"]
    # print(input_ids) 
    # position_ids = torch.tensor([[[0, 1, 2, 2], [0, 0, 0, 1]]]) 
    # attention_mask = torch.tensor(
    #     [[[
    #         [False, False, False, True],
    #         [False, False, False, True],
    #         [False, False, False, True],
    #         [False, False, False, False]     ]]] ) 
    # past_key_values = torch.zeros(28, 2, 0, 1, 32, 128) 
    # dynamic_axes={        
    #         "input_ids": {0: "batch_size", 1: "seq_len"},
    #         "position_ids": {0: "batch_size", 2: "seq_len"},
    #         "attention_mask": {0: "batch_size", 2: "seq_len", 3: "seq_len"},
    #         "past_key_values": {2: "history_len"} } 
    # torch.onnx.export(
    #     model,
    #     # f="onnx_output/chatglm_6b.onnx",
    #     f = onnx_path,
    #     args=(input_ids, position_ids, attention_mask, past_key_values),
    #     input_names=["input_ids", "position_ids", "attention_mask", "past_key_values"],
    #     output_names=["lm_logits"],
    #     dynamic_axes=dynamic_axes,
    #     opset_version=14, )

def CvtONNX2IR(onnx_path: str):
    print("Convert onnx model {} to IR.".format(onnx_path))
    cmd="mo -m ./onnx_output_cg/chat_glm_6b.onnx -o ./ir_output/"
    os.system(cmd)


# ConvertAutoModel(output_dir="./onnx_output_auto", output_onnx="chat_glm_6b.onnx")
ConvertCGModel(output_dir="./onnx_output_cg", output_onnx="chat_glm_6b.onnx")
# ConvertCGModel_myupdate(output_dir="./onnx_output_cg", output_onnx="chat_glm_6b.onnx")

# CvtONNX2IR("./onnx_output_cg/chat_glm_6b.onnx")
