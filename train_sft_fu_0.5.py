# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import torch
import json
import random
import numpy as np
from tqdm import tqdm
import datetime
tqdm.pandas()

import pickle
from transformers import pipeline, AutoTokenizer, set_seed, BertTokenizer
from datasets import load_dataset
from datasets import Dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead
from transformers import  AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from trl.core import LengthSampler
import tritonclient.http as httpclient
import time
########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# This example fine-tunes a GPT2 model on the IMDB dataset using PPO
# (proximal policy optimization).
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - multi GPUS (using DeepSpeed ZeRO-Offload stages 1 & 2)
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, first initialize the accelerate
# configuration with `accelerate config`
#
########################################################################
remote_ip='10.212.207.33:8000'
# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
# `accelerator_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.
class InputDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self
    def allowDotting(self, state=True):
        if state:
            self.__dict__ = self
        else:
            self.__dict__ = dict()


def GetRmBatch(prompt_list, response_list, RM_tokenizer, cur_device):
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    for prompt, response in zip(prompt_list, response_list):
        prompt = prompt.replace("<|startofpiece|>", "").replace("[回答]", "").replace("[CLS]", "").replace("\n", "").replace("<n>", "")
        response = response.replace("<|startofpiece|>", "").replace("<|endofpiece|>", "").replace("<|endoftext|>", "").replace("<n>", "##402").replace(" ","")
        RM_input = RM_tokenizer(prompt, response, truncation=True, max_length=1024, padding="max_length")
        input_ids_list.append(RM_input["input_ids"])
        attention_mask_list.append(RM_input["attention_mask"])
        token_type_ids_list.append(RM_input["token_type_ids"])
    result = InputDict([("input_ids", torch.tensor(input_ids_list).to(cur_device)),("attention_mask", torch.tensor(attention_mask_list).to(cur_device)),("token_type_ids", torch.tensor(token_type_ids_list).to(cur_device))])
    return result

def GetRmBatchNumpy(prompt_list, response_list, RM_tokenizer):
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    prompt_res = []
    prompt_res = []
    for prompt, response in zip(prompt_list, response_list):
        prompt = prompt.replace("<|startofpiece|>", "").replace("[CLS]", "").replace("\n", "<n>").replace("<|endoftext|>", "").strip("[gMASK]").strip("[回答]").strip()
        response = response.replace("<|startofpiece|>", "").replace("<|endofpiece|>", "").replace("<|endoftext|>", "").replace("[CLS]", "")
        new_prompt = prompt + "[SEP]" + response
        prompt_res.append(new_prompt)
        # RM_input = RM_tokenizer((prompt + "[UNUSED1]" + response)[:300], max_length=512, padding=True)
        # input_ids_list.append(RM_input["input_ids"])
        # attention_mask_list.append(RM_input["attention_mask"])
        # token_type_ids_list.append(RM_input["token_type_ids"])
    RM_input = RM_tokenizer(prompt_res, max_length=1024, padding=True, truncation=True, return_tensors="pt")
    # print('RM_input:', RM_input)
    result = [torch.tensor(RM_input["input_ids"]).numpy(),  torch.tensor(RM_input["attention_mask"]).numpy()]
    # result = InputDict([("input_ids", torch.tensor(input_ids_list).to(cur_device)),("attention_mask", torch.tensor(attention_mask_list).to(cur_device)),("token_type_ids", torch.tensor(token_type_ids_list).to(cur_device))])
    return result

class GLMPPOTrainer(PPOTrainer):
    def generate(self, inputs, gen_len):
        #response = self.accelerator.unwrap_model(self.model).generate(**inputs, max_length=512, eos_token_id=50007, num_beams=1, no_repeat_ngram_size=7, repetition_penalty=1.1, min_length=3)
        #response = self.accelerator.unwrap_model(self.model).generate(**inputs, max_new_tokens=gen_len, eos_token_id=50007, num_beams=1, no_repeat_ngram_size=7, repetition_penalty=1.1, min_length=3)
        response = self.accelerator.unwrap_model(self.model).generate(**inputs, max_new_tokens=400, eos_token_id=50007, top_k=0, top_p=1, do_sample=True, temperature=0.7)
        return response


config = PPOConfig(
    model_name="/search/ai/jamsluo/GLM_RLHF/sft_0.7",
    learning_rate=5e-6,
    batch_size=8,
    ppo_epochs=3,
    log_with="wandb",
    init_kl_coef=0.05,
    remove_unused_columns=False,
    mini_batch_size=8
)
#print(dir(config))
print(config.batch_size)
print(config.ppo_epochs)


# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.



class PPOIdxDataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.f = open("/search/ai/kaitongyang/ppo_glm_debug/data/bad_case_prompt/sft_data_fu_0.5/prompt.txt")
        with open("/search/ai/kaitongyang/ppo_glm_debug/data/bad_case_prompt/sft_data_fu_0.5/dataset_tmp.id", 'rb') as fp:
            self.offsets = pickle.load(fp)
    def __len__(self):
        return len(self.offsets)
    def __getitem__(self, index):
        self.f.seek(self.offsets[index], 0)
        cur_data = self.f.readline()
        # if len(cur_data) > 20 or len(cur_data) < 15:
        #     self.__getitem__(random.randint(0, len(self.offsets)))
        inputs = self.tokenizer(cur_data + " [回答][gMASK]", return_tensors="pt")
        for key in inputs:
            inputs[key] = inputs[key][:,:-1]
        inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=300)
        return inputs
       

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(0)
print("os LOCAL_RANK", os.environ["LOCAL_RANK"])
if int(os.environ["LOCAL_RANK"]) % 2 == 1:
    print("sleep some time" )
    time.sleep(80)

# Now let's build the model, the reference model, and the tokenizer.
time.sleep(int(os.environ["LOCAL_RANK"]))
tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name, trust_remote_code=True, remote_ip=remote_ip, triton_model_local="REL_sft_07")
# ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name, trust_remote_code=True)
model.set_tokenizer(tokenizer)
# ref_model.set_tokenizer(tokenizer)
print("start build dataset")
# dataset_path="/search/ai/kaitongyang/RLHF_DEBUG/RM/data/success-0223.json"
#dataset = build_dataset(dataset_path, tokenizer)
dataset = PPOIdxDataset(tokenizer)
#print(dataset)
print("num_dataset:"+str(len(dataset)))
# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = GLMPPOTrainer(config, model, None, tokenizer, dataset=dataset, data_collator=collator)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
no_update_device = "cuda:" + str(int(str(device).split(":")[-1])+4)
#no_update_device = "cuda:6"
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
'''
RM_model_path = "/search/ai/kaitongyang/RLHF_DEBUG/RM/summarization_reward_model/checkpoint-58"
RM_tokenizer = AutoTokenizer.from_pretrained(RM_model_path)
RM_model = AutoModelForSequenceClassification.from_pretrained(RM_model_path, num_labels=1)
RM_model.to(no_update_device)
'''
#
senti_tokenizer = BertTokenizer.from_pretrained('/search/ai/pretrain_models/cpt-large/', trust_remote_code=True)
# senti_model = AutoModelForSequenceClassification.from_pretrained('uer/roberta-base-finetuned-jd-binary-chinese')
# sentiment_pipe = pipeline('sentiment-analysis', model=senti_model, tokenizer=senti_tokenizer, device=no_update_device)
#triton_client = httpclient.InferenceServerClient(url=remote_ip, connection_timeout=300, network_timeout=300)
triton_client = httpclient.InferenceServerClient(url="10.212.207.33:12356", connection_timeout=300, network_timeout=300)
# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "max_length":512,
    "eos_token_id":tokenizer.eop_token_id,
    "num_beams":1,
    "no_repeat_ngram_size":7,
    "repetition_penalty":1.1,
    "min_length":3,
}
output_min_length = 20
output_max_length = 40
output_length_sampler = LengthSampler(output_min_length, output_max_length)
#print("*"*10)
#print(len(ppo_trainer.dataloader))
#print("="*10)
for cur_big_epoch in range(10):
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_input_ids_tensors = batch["input_ids"]
        query_input_ids_tensors = list(query_input_ids_tensors)
        texts = []
        for i in range(len(query_input_ids_tensors)):
            cur_text = tokenizer.decode(query_input_ids_tensors[i][0][1:-1])
            texts.append(cur_text)
        query_tensor = tokenizer(texts, padding=True, return_tensors="pt")
        # for key in query_tensor:
        #     query_tensor[key] = query_tensor[key][:, :-1]
        query_tensor = tokenizer.build_inputs_for_generation(query_tensor, max_gen_length=512)
        query_tensor.to(device)
        if str(ppo_trainer.accelerator.device) == "cuda:0":
            print(str(cur_big_epoch) + "epoch:"+str(epoch) + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        gen_len = output_length_sampler()
        response = ppo_trainer.generate(query_tensor, gen_len)[:, query_tensor["input_ids"].size()[1]:]
        padding_mask = response!=50007
        padding_mask = padding_mask.int()
        response_tensor_temp = [i[:sum(j)+1] for i,j in zip(response.tolist(), padding_mask.tolist())]
        response_tensor = []
        for temp_logit in response_tensor_temp:
            cur_response_tensor = []
            for cur_id in temp_logit:
                if int(cur_id) >= 50010:
                    continue
                else:
                    cur_response_tensor.append(cur_id)
            assert len(cur_response_tensor) > 0
            response_tensor.append(torch.tensor(cur_response_tensor))
        batch["query"] = [tokenizer.decode(r) for r in query_tensor["input_ids"].tolist()]
        batch["response"] = [tokenizer.decode(logits) for logits in response_tensor]
        if str(ppo_trainer.accelerator.device) == "cuda:0":
            for i,j in zip(batch["query"], batch["response"]):
                print("*"*6)
                print(i)
                print("="*3)
                print(j)
        if str(ppo_trainer.accelerator.device) == "cuda:0":
            print("inference: " +  str(sum([len(i) for i  in batch["response"]])) + "time :" +datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        #### Compute sentiment score
        '''
        RM_batch = GetRmBatch(batch["query"], batch["response"], RM_tokenizer, no_update_device)
        rewards = RM_model(input_ids=RM_batch["input_ids"], attention_mask=RM_batch["attention_mask"], token_type_ids=RM_batch["token_type_ids"])
        rewards = [torch.tensor(reward) for reward in rewards["logits"].cpu().tolist()]
        '''
        rewards = []
        RM_batch = GetRmBatchNumpy(batch["query"], batch["response"], senti_tokenizer)
        inputs = []
        print("RM shape: ", RM_batch[0].shape, RM_batch[1].shape)
        inputs.append(httpclient.InferInput('input_ids', list(RM_batch[0].shape), 'INT64'))
        inputs.append(httpclient.InferInput('attention_mask', list(RM_batch[1].shape), 'INT64'))
        inputs[0].set_data_from_numpy(RM_batch[0])
        inputs[1].set_data_from_numpy(RM_batch[1])
        output = httpclient.InferRequestedOutput('output')
        # try:
        results = triton_client.infer(
            "RM_cpt_onnx",
            inputs,
            model_version='1',
            outputs=[output],
            request_id='1',
            timeout=300 * 1000
        )
        results = results.as_numpy('output')
        # rewards = []
        # for rsp in batch["response"]:
        #     tmp_score = 1.5*(rsp.count('<n>') - 5)
        #     rewards.append(torch.tensor(tmp_score))
            # if len(rsp) < 100:
            #     rewards.append(torch.tensor(-5.))
            # else:
            #     rewards.append(torch.tensor(5.))
        print(results)
        rewards = [torch.tensor(results[i][1]) for i in range(len(results))]
        # except:
        #     rewards = [torch.tensor(0.)]*config.batch_size
        #print(rewards)
        if str(ppo_trainer.accelerator.device) == "cuda:0":
            print(str(ppo_trainer.accelerator.device))
            print("RM time : " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        stats = ppo_trainer.step([torch.tensor(r).squeeze() for r in query_tensor["input_ids"].tolist()], response_tensor, rewards)
        if str(ppo_trainer.accelerator.device) == "cuda:0":
            print(str(ppo_trainer.accelerator.device))
            print("ppo trainer time : " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        ppo_trainer.log_stats(stats, batch, rewards)

        if epoch % 30 == 0 and str(ppo_trainer.accelerator.device) == "cuda:0":
            reward_path = "/search/ai/jamsluo/GLM_RLHF/ppo_glm/sft_fu_0.5"
            root_path = os.path.join(reward_path, str(cur_big_epoch) + "_" + str(epoch))
            if os.path.exists(root_path):
                pass
            else:
                os.mkdir(root_path)
            model.save_pretrained(root_path)
            tokenizer.save_pretrained(root_path)
    print("=="*20)
    print(cur_big_epoch)
    print("successfully!!!")
print("all succ")
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
