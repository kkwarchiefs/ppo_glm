from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#path = "/search/ai/kaitongyang/GLM_RLHF/PPO_trl/glm_0.3"
path = "/search/ai/kaitongyang/RLHF_DEBUG/PPO_trl/small_glm"
#path = "/search/ai/kaitongyang/RLHF_DEBUG/PPO_trl/RLHF_MODEL_small_glm_fb16/9_60"
path = "/search/ai/kaitongyang/RLHF_DEBUG/PPO_trl/RLHF_MODEL_big_glm_fb16/0_200"
device = "cuda:3"
suffix = "[gMASK]"

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(path, trust_remote_code=True)
model = model.half().to(device)
model.eval()

print("load model and tokenizer done !!!")


# Inference
text_list = ["房间非常大，", "小本子不错，", "性价比高，", "比较老的酒店，", "屏幕小了一点，", "地理位置不错", "配件太少了", "收到本书，非常兴奋，", "外观漂亮，", "机器不错。"]


for text in text_list:
    inputs = tokenizer(text + suffix, return_tensors="pt")
    for key in inputs:
        inputs[key] = inputs[key][:,:-1]
    inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)
    inputs = inputs.to(device)
    outputs = model.generate(**inputs, eos_token_id=tokenizer.eop_token_id, max_length=512, min_length=-1, top_k=0.0, top_p=1.0, do_sample=True)
    print("*"*10)
    print(text)
    print("="*5)
    print(tokenizer.decode(outputs.squeeze().tolist()[inputs["input_ids"].size()[1]:]).replace("\"", ""))
