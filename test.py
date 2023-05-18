from transformers import AutoTokenizer, AutoModel
import time
model_path="THUDM/chatglm-6b"
model_path="./local_chat_glm_6b_model"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
# response, history = model.chat(tokenizer, "你好", history=[])
# print(response)
# response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
# print(response)
t0=time.time()
response, history = model.chat(tokenizer, "教我做西班牙风情牛肉饭", history=[])
t2=time.time()
print(response)
print("time =", (t2-t0))