from local_chat_glm_6b_model.tokenization_chatglm import ChatGLMTokenizer
from local_chat_glm_6b_model.modeling_chatglm import ChatGLMForConditionalGeneration

tokenizer = ChatGLMTokenizer.from_pretrained("./local_chat_glm_6b_model", trust_remote_code=True)
# model = ChatGLMForConditionalGeneration.from_pretrained("./local_chat_glm_6b_model", trust_remote_code=True).half().cuda()
model = ChatGLMForConditionalGeneration.from_pretrained("./local_chat_glm_6b_model", trust_remote_code=True).float().cpu()

response, history = model.chat(tokenizer, "你好", history=[])
print(response)

# response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
# print(response)