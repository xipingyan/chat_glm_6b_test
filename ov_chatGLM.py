import os
from io import BytesIO
import torch
from torch import nn
import time 
import argparse
#from local_chat_glm_6b_model.configuration_chatglm import ChatGLMConfig
from local_chat_glm_6b_model.modeling_chatglm import ChatGLMForConditionalGeneration #ChatGLMModel
from local_chat_glm_6b_model.tokenization_chatglm import ChatGLMTokenizer
from transformers import AutoTokenizer, AutoModel
import numpy as np
import onnxruntime as ort
from openvino.runtime import Core
from openvino.runtime import compile_model
# from openvino.tools.mo import convert_model
from transformers.modeling_outputs import (
    CausalLMOutputWithPast
)
from transformers.generation.utils import LogitsProcessorList
#from transformers.onnx import export, FeaturesManager

# import subprocess
# subprocess.Popen(["/bin/sh", "source ../openvino/build/install/setupvars.sh"])

model_path="./local_chat_glm_6b_model"
tokenizer = ChatGLMTokenizer.from_pretrained(model_path)
bos_token_id=tokenizer.bos_token_id
eos_token_id=tokenizer.eos_token_id
mask_token_id=tokenizer.mask_token_id
position_encoding_2d = True

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

def update_inputs_for_generation(
    inputs) -> dict:

    attention_mask = torch.from_numpy(inputs["attention_mask"])

    if attention_mask is not None and attention_mask.dtype == torch.bool:
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones((*attention_mask.shape[:3], 1))], dim=3)
        new_attention_mask = attention_mask[:, :, -1:].clone()
        new_attention_mask[..., -1] = False
        inputs["attention_mask"] = torch.cat([attention_mask, new_attention_mask], dim=2).numpy()
    
    position_ids = torch.from_numpy(inputs["position_ids"])
    new_position_id = position_ids[..., -1:].clone()
    new_position_id[:, 1, :] += 1
    inputs["position_ids"] = torch.cat(
        [position_ids, new_position_id], dim=-1
    ).numpy()

    return inputs

    

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

def generate_sequence(engine,inputs,max_sequence_length=128,
                      eos_token_id=eos_token_id, dynamic_shapes=True):
    unfinished_sequences = torch.from_numpy(inputs['input_ids']).new(inputs['input_ids'].shape[0]).fill_(1)
    while True:
        #cur_input_len = len(inputs['input_ids'][0])
        if engine == "OV":            
            print("Start infer_new_request.")
            outputs = compiled_model.infer_new_request(inputs)
            next_token_logits = outputs["lm_logits"][:, -1, :]
        elif engine == "ORT":
            names = [o.name for o in ort_sess._sess.outputs_meta]
            outputs = ort_sess.run(input_feed=inputs,output_names=names)
            '''print(f"==output shape of logits:{outputs[0].shape}")
            print(f"==output shape of past:{outputs[1].shape}")
            print(f"=============================")'''
            next_token_logits = outputs[0][:, -1, :]
        else:
            break
        #next_token_logits = outputs["lm_logits"][:, -1, :]
        logits_processor = LogitsProcessorList()
        next_token_scores = logits_processor(torch.from_numpy(inputs['input_ids']), next_token_logits)
        #next_token_scores = logits_warper(inputs['input_ids'], next_token_scores)
        probs = nn.functional.softmax(torch.from_numpy(next_token_scores), dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in [eos_token_id])).long())

        if unfinished_sequences.max() == 0:
            break
        else:
            inputs['input_ids'] = torch.cat([torch.from_numpy(inputs['input_ids']), next_tokens[:, None]], dim=-1).numpy()
            #update pas_key_values
            '''inputs['past_key_values'] = np.zeros((28,2,inputs['input_ids'].shape[1],1,32,128),dtype=np.float32) # shape [28,2,?,1,32,128]
            past_key_values_pair = np.zeros((2,inputs['input_ids'].shape[1],1,32,128),dtype=np.float32) #pair shape [2,?,1,32,128]
            #inputs['past_key_values'] = np.zeros((28,2,inputs['input_ids'].shape[1]-1,1,32,128),dtype=np.float32)
            #past_key_values_pair = np.zeros((2,inputs['input_ids'].shape[1]-1,1,32,128),dtype=np.float32)
            for i in range(len(outputs)):
                if i==0 or i%2==1:
                    continue
                elif i%2==0:
                    past_key_values_pair = np.stack((np.pad(outputs[i-1],((0,1),(0,0),(0,0),(0,0))),np.pad(outputs[i],((0,1),(0,0),(0,0),(0,0)))),axis=0) # i-1 is key, i is values
                    #past_key_values_pair = np.stack((outputs[i-1],outputs[i]),axis=0)
                    inputs['past_key_values'][int(i/2)-1] = past_key_values_pair
            '''
            inputs = update_inputs_for_generation(inputs)
            inputs["past_key_values"] = np.zeros((28,2,0,1,32,128),dtype=np.float32)
            '''print(f"==input shape of input_ids:{inputs['input_ids'].shape}")
            print(f"==input shape of position_ids:{inputs['position_ids'].shape}")
            print(f"==input shape of attention_mask:{inputs['attention_mask'].shape}")
            print(f"==input shape of past_key_values:{inputs['past_key_values'].shape}")'''
    return inputs['input_ids']

'''def stream_chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 100,
                do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
    if history is None:
        history = []
    if logits_processor is None:
        logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    gen_kwargs = {"max_length": max_length, "do_sample": do_sample, "top_p": top_p,
                    "temperature": temperature, "logits_processor": logits_processor, **kwargs}
    if not history:
        prompt = query
    else:
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
        prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = inputs.to('cpu')
    #for outputs in self.stream_generate(**inputs, **gen_kwargs):
    for outputs in generate_sequence(engine,inputs):
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        response = self.process_response(response)
        new_history = history + [(query, response)]
        yield response, new_history'''

parser = argparse.ArgumentParser()
parser.add_argument("-e","--engine", type=str, default="OV", help="Specify inference engine (OV, ORT, Torch) to inference. Default is OV")
parser.add_argument("-p","--prompt", type=str, default="你好", help="Specify input prompt. Default is 你好")
args = parser.parse_args()

#====input prompt process=====
inputs = {}
input_text = args.prompt
input_token = tokenizer([input_text], return_tensors="np")["input_ids"]
inputs['input_ids'] = input_token
token_size = inputs['input_ids'].shape[1]
print(f"==Input tokens:{inputs['input_ids']}")
inputs = prepare_inputs_for_generation(input_ids=torch.from_numpy(inputs['input_ids']))
'''print(f"==input shape of input_ids:{inputs['input_ids'].shape}")
print(f"==input shape of position_ids:{inputs['position_ids'].shape}")
print(f"==input shape of attention_mask:{inputs['attention_mask'].shape}")
print(f"==input shape of past_key_values:{inputs['past_key_values'].shape}")'''

read_end = 0
compile_end = 0

if args.engine == "OV":
    ov_model_path = 'ov_output/chat_glm_6b.xml'
    core = Core()
    # Auto cache and loading.
    core.set_property({'CACHE_DIR': './cache_ov'})

    print("Start read_model. ", ov_model_path)
    read_start = time.time()
    model = core.read_model(ov_model_path)
    read_end = time.time()

    compiled_model = core.compile_model(model, "CPU")
    compile_end = time.time()
elif args.engine == "ORT":
    ort_model_path = 'onnx_output_cg/chat_glm_6b.onnx'
    read_start = time.time()
    ort_sess = ort.InferenceSession(ort_model_path,providers=['CPUExecutionProvider','OpenVINOExecutionProvider'])
    read_end = time.time()
    compile_end = read_end
else:
    read_start = time.time()
    model = ChatGLMForConditionalGeneration.from_pretrained(model_path).float().cpu()
    model = model.eval()
    read_end = time.time()
    compile_end = read_end

output_ids = generate_sequence(args.engine,inputs)
#inp_tokenizer = tokenizer(args.prompt, return_tensors="pt").to("cpu")
#output_ids = model.generate(**inp_tokenizer, do_sample=True)
infer_latency = time.time() -compile_end

output_text = tokenizer.decode(output_ids[0])

generate_tokens = len(output_ids[0]) - len(input_token[0])

print(f"Predicted Sequence:{output_text}")

print('===Model read spend time: %.1f secondes==='%(read_end-read_start))
print('===Model compile spend time: %.1f secondes==='%(compile_end-read_end))
print('===Infer generation spend time: %.1f seconds==='%(infer_latency))
print('===generation time per token %.3f s/token==='%(infer_latency/generate_tokens))

#print(inputs['past_key_values'])



'''prompt="Hello chatGLM"
inp_tokenizer = tokenizer(prompt, return_tensors="pt").to(device)
print(inp_tokenizer)

generated_ids = model.generate(**inp_tokenizer, do_sample=True)
output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(output)

#ov_model = convert_model(model,example_input=inp_tokenizer)
#compiled_model = compile_model(ov_model)
#result = compiled_model(inp_tokenizer)
print("ok")'''