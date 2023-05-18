
import onnxruntime as ort
import numpy as np
import onnx
import time
import torch

from local_chat_glm_6b_model.tokenization_chatglm import ChatGLMTokenizer
model_path="./local_chat_glm_6b_model"
tokenizer = ChatGLMTokenizer.from_pretrained(model_path) 

bos_token_id=tokenizer.bos_token_id
eos_token_id=tokenizer.eos_token_id
mask_token_id=tokenizer.mask_token_id
position_encoding_2d = True

print("--> Check model.")
onnx_model = 'onnx_output_auto/chat_glm_6b.onnx'
onnx_model = 'onnx_output_cg/chat_glm_6b.onnx'
onnx.checker.check_model(onnx_model)

print("--> ort.InferenceSession.")
read_start = time.time()
# ort_sess = ort.InferenceSession(onnx_model, providers=['CPUExecutionProvider','OpenVINOExecutionProvider'])
ort_sess = ort.InferenceSession(onnx_model)
read_end = time.time()

def get_position_ids(input_ids, mask_positions, device, use_gmasks=None):
    batch_size, seq_length = input_ids.shape
    if use_gmasks is None:
        use_gmasks = [False] * batch_size
    context_lengths = [seq.tolist().index(bos_token_id) for seq in input_ids]
    if position_encoding_2d:
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
        for i, context_length in enumerate(context_lengths):
            position_ids[i, context_length:] = mask_positions[i]
        block_position_ids = [torch.cat((
            torch.zeros(context_length, dtype=torch.long, device=device),
            torch.arange(seq_length - context_length, dtype=torch.long, device=device) + 1
        )) for context_length in context_lengths]
        block_position_ids = torch.stack(block_position_ids, dim=0)
        position_ids = torch.stack((position_ids, block_position_ids), dim=1)
    else:
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
        for i, context_length in enumerate(context_lengths):
            if not use_gmasks[i]:
                position_ids[context_length:] = mask_positions[i]

    return position_ids

def get_masks(input_ids, device):
    batch_size, seq_length = input_ids.shape
    context_lengths = [seq.tolist().index(bos_token_id) for seq in input_ids]
    attention_mask = torch.ones((batch_size, seq_length, seq_length), device=device)
    attention_mask.tril_()
    for i, context_length in enumerate(context_lengths):
        attention_mask[i, :, :context_length] = 1
    attention_mask.unsqueeze_(1)
    attention_mask = (attention_mask < 0.5).bool()

    return attention_mask

def prepare_inputs_for_generation(
    input_ids,
    bos_token_id=bos_token_id,
    mask_token_id=mask_token_id,
    past=None,
    past_key_values=None,
    attention_mask=None,
    position_ids = None) -> dict:

    batch_size, seq_length = input_ids.shape
    MASK, gMASK = mask_token_id, mask_token_id+1
    seqs = input_ids.tolist()
    mask_positions, use_gmasks = [], []
    for seq in seqs:
        mask_token = gMASK if gMASK in seq else MASK
        use_gmask = mask_token == gMASK
        mask_positions.append(seq.index(mask_token))
        use_gmasks.append(use_gmask)

    # only last token for input_ids if past is not None
    if past is not None or past_key_values is not None:
        last_token = input_ids[:, -1].unsqueeze(-1)
        if attention_mask is not None and attention_mask.dtype == torch.bool:
            attention_mask = attention_mask[:, :, -1:]
        else:
            attention_mask = None
        if position_ids is not None:
            position_ids = position_ids[..., -1:]
        else:
            context_lengths = [seq.index(bos_token_id) for seq in seqs]
            if position_encoding_2d:
                position_ids = torch.tensor(
                    [[mask_position, seq_length - context_length] for mask_position, context_length in
                        zip(mask_positions, context_lengths)], dtype=torch.long, device="cpu").unsqueeze(-1)
            else:
                position_ids = torch.tensor([mask_position for mask_position in mask_positions], dtype=torch.long,
                                            device="cpu").unsqueeze(-1)

        if past is None:
            past = past_key_values
        return {
            "input_ids": input_ids.numpy(),
            "position_ids": position_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
            "past_key_values": past.numpy()
        }
    else:
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            attention_mask = None
        if attention_mask is None:
            attention_mask = get_masks(
                input_ids,
                device="cpu"
            )
        if position_ids is None:
            position_ids = get_position_ids(
                input_ids,
                device="cpu",
                mask_positions=mask_positions,
                use_gmasks=use_gmasks
            )
        if past is None:
            past = torch.zeros(28, 2, 0, 1, 32, 128)
        return {
            "input_ids": input_ids.numpy(),
            "position_ids": position_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
            "past_key_values": past.numpy()
        }
          

# # Load the ONNX model
# print("--> Load the ONNX model.")
# model = onnx.load(onnx_model)

# # Print a human readable representation of the graph
# print("--> printable_graph.")
# print(onnx.helper.printable_graph(model.graph))

# print("--> ort.InferenceSession.")
# read_start = time.time()
# ort_sess = ort.InferenceSession(onnx_model,providers=['CPUExecutionProvider','OpenVINOExecutionProvider'])
# read_end = time.time()

# ort_sess = ort.InferenceSession()
# test = "晚上睡不着应该怎么办"

inputs = {}
input_text = "你好"
input_token = tokenizer([input_text], return_tensors="np")["input_ids"]
inputs['input_ids'] = input_token
token_size = inputs['input_ids'].shape[1]
print(f"==Input tokens:{inputs['input_ids']}")

inputs = prepare_inputs_for_generation(input_ids=torch.from_numpy(inputs['input_ids']))
print(f"==input shape of input_ids:{inputs['input_ids'].shape}")
print(f"==input shape of position_ids:{inputs['position_ids'].shape}")
print(f"==input shape of attention_mask:{inputs['attention_mask'].shape}")
print(f"==input shape of past_key_values:{inputs['past_key_values'].shape}")

names = [o.name for o in ort_sess._sess.outputs_meta]
outputs = ort_sess.run(input_feed=inputs,output_names=names)
print(f"==output shape of logits:{outputs[0].shape}")
print(f"==output shape of past:{outputs[1].shape}")
print(f"=============================")
# next_token_logits = outputs[0][:, -1, :]
