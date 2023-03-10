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

from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from datasets import Dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead
from transformers import  AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from trl.core import LengthSampler

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
        response = response.replace("<|startofpiece|>", "").replace("<|endofpiece|>", "").replace("<|endoftext|>", "").replace("<n>", "").replace(" ","")
        RM_input = RM_tokenizer(prompt, response, truncation=True, max_length=1024, padding="max_length")
        input_ids_list.append(RM_input["input_ids"])
        attention_mask_list.append(RM_input["attention_mask"])
        token_type_ids_list.append(RM_input["token_type_ids"])
    result = InputDict([("input_ids", torch.tensor(input_ids_list).to(cur_device)),("attention_mask", torch.tensor(attention_mask_list).to(cur_device)),("token_type_ids", torch.tensor(token_type_ids_list).to(cur_device))])
    return result


class GLMPPOTrainer(PPOTrainer):
    def generate(self, inputs):
        response = self.accelerator.unwrap_model(self.model).generate(**inputs, max_length=512, eos_token_id=50007, num_beams=1, no_repeat_ngram_size=7, repetition_penalty=1.1, min_length=3)
        return response


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, and `torch`.
    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


config = PPOConfig(
    model_name="/search/ai/kaitongyang/RLHF_DEBUG/PPO_trl/cur_model/30630",
    learning_rate=1.41e-5,
    batch_size=16,
    ppo_epochs=1,
    remove_unused_columns=False,
    mini_batch_size=8
)
#print(dir(config))
print(config.batch_size)
print(config.ppo_epochs)


# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.


def build_dataset(data_path, tokenizer):
    datas = open(data_path).read().splitlines()
    input_ids_list = []
    position_ids_list = []
    generation_attention_mask_list = []
    for data in datas[:1000]:
        data = json.loads(data)
        prompt = data["prompt"]
        inputs = tokenizer(prompt+" [回答]"+"[gMASK]", return_tensors="pt")
        for key in inputs:
            inputs[key] = inputs[key][:,:-1]
        inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)
        input_ids_list.append(inputs["input_ids"])
        position_ids_list.append(inputs["position_ids"])
        generation_attention_mask_list.append(inputs["generation_attention_mask"]) 
    result = Dataset.from_dict({"input_ids": input_ids_list, "position_ids":position_ids_list, "generation_attention_mask":generation_attention_mask_list})
    return result


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(0)

# Now let's build the model, the reference model, and the tokenizer.
tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name, trust_remote_code=True)
ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name, trust_remote_code=True)
model.set_tokenizer(tokenizer)
ref_model.set_tokenizer(tokenizer)
print("start build dataset")
dataset_path="/search/ai/kaitongyang/RLHF_DEBUG/RM/data/success-0223.json"
dataset = build_dataset(dataset_path, tokenizer)
print(dataset)
print("num_dataset:"+str(len(dataset)))
# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = GLMPPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
no_update_device = "cuda:" + str(int(str(device).split(":")[-1])+4)
#no_update_device = "cuda:6"
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

RM_model_path = "/search/ai/kaitongyang/RLHF_DEBUG/RM/summarization_reward_model/checkpoint-58"
RM_tokenizer = AutoTokenizer.from_pretrained(RM_model_path)
RM_model = AutoModelForSequenceClassification.from_pretrained(RM_model_path, num_labels=1)
RM_model.to(no_update_device)


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
output_min_length = 50
output_max_length = 150 
output_length_sampler = LengthSampler(output_min_length, output_max_length)
#print("*"*10)
#print(len(ppo_trainer.dataloader))
#print("="*10)
for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_input_ids_tensors = batch["input_ids"]
    query_position_ids_tensors = batch["position_ids"]
    query_generation_attention_mask_tensors = batch["generation_attention_mask"]
    if str(ppo_trainer.accelerator.device) == "cuda:0":    
        print("epoch:"+str(epoch) + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    response_tensors = []
    for query_input_ids, query_position_ids, query_generation_attention_mask in zip(query_input_ids_tensors, query_position_ids_tensors, query_generation_attention_mask_tensors):
        query = InputDict([("input_ids", torch.tensor(query_input_ids).to(device)), ("position_ids", torch.tensor(query_position_ids).to(device)), ("generation_attention_mask", torch.tensor(query_generation_attention_mask).to(device))])    
        gen_len = output_length_sampler()
        response = ppo_trainer.generate(query)
        response = response.squeeze().cpu().tolist()[len(query_input_ids[0]):][-gen_len:]
        #response = response + [50000] * (gen_len-len(response))
        response_tensors.append(torch.tensor(response))
    batch["prompt"] = [tokenizer.decode(torch.tensor(r).squeeze()) for r in query_input_ids_tensors]
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
    #print(batch["prompt"])
    #print(batch["response"])
    if str(ppo_trainer.accelerator.device) == "cuda:0":
        print("inference: " +  str(sum([len(i) for i  in batch["response"]])) + "time :" +datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    #### Compute sentiment score
    RM_batch = GetRmBatch(batch["prompt"], batch["response"], RM_tokenizer, no_update_device)
    rewards = RM_model(input_ids=RM_batch["input_ids"], attention_mask=RM_batch["attention_mask"], token_type_ids=RM_batch["token_type_ids"])
    rewards = [torch.tensor(reward) for reward in rewards["logits"].cpu().tolist()]
    if str(ppo_trainer.accelerator.device) == "cuda:0":
        print(str(ppo_trainer.accelerator.device))
        print("RM time : " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
     
    stats = ppo_trainer.step([torch.tensor(r).squeeze() for r in query_input_ids_tensors], response_tensors, rewards)
    if str(ppo_trainer.accelerator.device) == "cuda:0":
        print(str(ppo_trainer.accelerator.device))
        print("ppo trainer time : " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    ppo_trainer.log_stats(stats, batch, rewards)
    
    if epoch%20 == 0 and str(ppo_trainer.accelerator.device) == "cuda:0":
        reward_path = "/search/ai/kaitongyang/RLHF_DEBUG/PPO_trl/RLHF_MODEL"
        root_path = os.path.join(reward_path, str(epoch))
        if os.path.exists(root_path):
            pass
        else:
            os.mkdir(root_path)
        model.save_pretrained(root_path)
        tokenizer.save_pretrained(root_path)
print("all succ")
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
