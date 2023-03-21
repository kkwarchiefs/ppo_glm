from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
import datetime
import torch

path = "/search/ai/kaitongyang/online/model/GLM-10B-chinese-customization_03-07-21-23"
path = "/search/ai/kaitongyang/RLHF_DEBUG/PPO_trl/glm_0.5"
# path = '/search/ai/pretrain_models/glm-large-chinese/'
# path = '/search/ai/kaitongyang/ppo_glm_debug/RLHF_MODEL_big_glm_fb16_beam_e6/0_340'
device = "cuda:6"
suffix = "[回答][gMASK]"
set_seed(0)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(path, trust_remote_code=True)
model = model.half().to(device)
model.eval()

print("load model and tokenizer done !!!")
def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = torch.nn.functional.log_softmax(logits, dim=2)
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy

def get_max_idx(logits):
    return torch.argmax(logits,dim=2)
# Inference
text_list = ["去美国旅游，推荐一些景点", "北京有哪些好吃的饭馆", "推荐几款洗面奶", "给我写一首 北京好美的七言绝句"]

while True:
    raw_text = input("\nContext prompt (stop to exit) >>> ")
    if not raw_text:
        print('Prompt should not be empty!')
        continue
    if raw_text == "stop":
        terminate_runs = 1
        break
    texts = raw_text.split('|')
    texts = [a + suffix for a in texts]
    # if len(items) != 4:
    #     print('param should not be empty!')
    #     continue
    # text = items[0]
    # texts = [text + suffix, text[:-2] + suffix, text[:-4] + suffix, text[:-6] + suffix]
    # top_k = int(items[1])
    # top_p = float(items[2])
    # repetition_penalty = float(items[3])
    inputs_ori = tokenizer(texts, padding=True, return_tensors="pt")
    inputs_ori = inputs_ori.to(device)
    for key in inputs_ori:
        inputs_ori[key] = inputs_ori[key][:, :-1]
    inputs = tokenizer.build_inputs_for_generation(inputs_ori, max_gen_length=512)
    # inputs = inputs.to(device)
    outputs = model.generate(**inputs, max_new_tokens=256, eos_token_id=50007, top_k=20, top_p=0.6,repetition_penalty=1.3, do_sample=False)
    # response_text = [tokenizer.decode(logits) for logits in outputs.tolist()]
    # print(response_text)
    response_text = [tokenizer.decode(logits) for logits in outputs[:, inputs["input_ids"].size()[1]:].tolist()]
    print(response_text)
    # temp_inputs = tokenizer.build_inputs_for_generation(inputs_ori, targets=response_text, max_gen_length=256, padding=False)
    # # print(temp_inputs)
    # base_model_output = model(**temp_inputs)
    # idx = get_max_idx(base_model_output.logits)
    # print([tokenizer.decode(logits) for logits in idx.tolist()])
    # print(inputs_ori["input_ids"])
    # print(temp_inputs["input_ids"])
    # print(idx)
    # # last_hidden_state = base_model_output.loss[:,inputs_ori.size()[1]-1:-1,:]
    # #last_hidden_state = base_model_output.mems[-1]
    # #print(last_hidden_state.size())
    # print("logits.shape", base_model_output.logits.shape)
    # print("temp_inputs", temp_inputs["input_ids"].shape)
    # lm_logits = base_model_output.logits[:,inputs_ori["input_ids"].size()[1]:-1,:]
    # cur_input_ids = temp_inputs["input_ids"][:,inputs_ori["input_ids"].size()[1]+1:]
    # print("lm_logits", lm_logits.shape)
    # print("cur_input_ids", cur_input_ids.shape)
    # print(logprobs_from_logits(lm_logits, cur_input_ids))
    # lm_logits = base_model_output.logits[:,inputs_ori["input_ids"].size()[1]:,:]
    # cur_input_ids = temp_inputs["input_ids"][:,inputs_ori["input_ids"].size()[1]:]
    # print(cur_input_ids)
    # print(logprobs_from_logits(lm_logits, cur_input_ids))
    # lm_logits = base_model_output.logits[:,inputs_ori["input_ids"].size()[1]+1:,:]
    # cur_input_ids = temp_inputs["input_ids"][:,inputs_ori["input_ids"].size()[1]:-1]
    # print(cur_input_ids)
    # print(logprobs_from_logits(lm_logits, cur_input_ids))
    # # cur_input_ids_text = [tokenizer.decode(logits) for logits in cur_input_ids.tolist()]
    # # print(cur_input_ids_text)
    # # print(tokenizer.decode(outputs.squeeze().tolist()))
    # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

