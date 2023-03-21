from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import datetime
import time

path = "/group/20018/shahmei/model/GLM-10B-chinese-customization_02-28-11-26/30630"
#path = "/search/ai/kaitongyang/online/model/GLM-10B-chinese-customization_02-28-11-26/30630"
#path = "/search/ai/kaitongyang/RLHF_DEBUG/PPO_trl/cur_model/30630"
#path = "/search/ai/kaitongyang/RLHF_DEBUG/PPO_trl/RLHF_MODEL/100"
#path = "/search/ai/kaitongyang/RLHF_DEBUG/PPO_trl/RLHF_MODEL_PRE/60"
device = "cuda:0"
suffix = " [回答][gMASK]"

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(path, trust_remote_code=True)
model = model.half().to(device)
model.eval()

#print("load model and tokenizer done !!!")

#print("=====")

#print(model)

#exit(0)

# Inference
text_list = ["如何清理手机内存?", "新能源汽车推荐买哪个", "家长对幼儿园孩子情况评价怎么写?", "中国女生与外国的女生有什么区别?", "有哪些网站,一旦知道,你就离>    不开了，并说明为什么", "流浪地球观后感"]

for text in text_list:
    inputs = tokenizer(text + suffix, return_tensors="pt")
    for key in inputs:
        inputs[key] = inputs[key][:,:-1]
    inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)
    inputs = inputs.to(device)
    start_time = datetime.datetime.now()
    #print(start_time.strftime('%Y-%m-%d %H:%M:%S'))
    #outputs = model.generate(**inputs, max_new_tokens=512, eos_token_id=tokenizer.eop_token_id, num_beams=4, no_repeat_ngram_size=7, repetition_penalty=1.1, min_length=3)
    #outputs = model.generate(**inputs, max_length=256, eos_token_id=tokenizer.eop_token_id, top_k=20, top_p=0.6,
    #                         repetition_penalty=1.3)
    outputs = model.generate(**inputs, max_length=256, eos_token_id=tokenizer.eop_token_id, top_k=20, top_p=0.6,
                             repetition_penalty=1.3, do_sample=False)


    start_time = datetime.datetime.now()
    #for _ in range(50):
    #    outputs = model.generate(**inputs, max_length=256, eos_token_id=tokenizer.eop_token_id, top_k=0, top_p=0.6,repetition_penalty=1.3, do_sample=False)
    second_time = datetime.datetime.now()
    #for key in inputs:
    #    print(key, inputs[key].shape)
    #print("output", outputs.shape)
    print("\n\ngenerate step:" + second_time.strftime('%Y-%m-%d %H:%M:%S')+",take:"+str((second_time-start_time)/50)+"\n\n")

    print(tokenizer.decode(outputs.squeeze().tolist()))
    #third_time = datetime.datetime.now()
    #print("decode step:" + third_time.strftime('%Y-%m-%d %H:%M:%S')+",take:"+str(third_time-second_time))
