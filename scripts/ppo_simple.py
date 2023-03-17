"""
@Date   ：2022/11/2
@Author ：
"""
import random
import gym
import torch
import numpy as np
from matplotlib import pyplot as plt
from IPython import display

env = gym.make("CartPole-v0")
# 智能体状态
state = env.reset()
# 动作空间
actions = env.action_space.n
print(state, actions)

# 演员模型：接收一个状态，输出一个动作策略的概率分布，智能体采用执行动作
actor_model = torch.nn.Sequential(torch.nn.Linear(4, 128),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(128, 2),
                                  torch.nn.Softmax())
# 评论员模型：评价一个状态的价值，给出多好的得分
critic_model = torch.nn.Sequential(torch.nn.Linear(4, 128),
                                   torch.nn.ReLU(),
                                   torch.nn.Linear(128, 1))
# 演员模型执行一个动作（采样获得）
def get_action(state):
    state = torch.FloatTensor(state).reshape(1, 4)
    probs = actor_model(state)
    action = random.choices(range(2), weights=probs[0], k=1)[0]

    return action

# 获取一个回合的样本数据
def get_data():
    states = []
    rewards = []
    actions = []
    next_states = []
    dones = []

    state = env.reset()
    done = False
    while not done:
        action = get_action(state)
        next_state, reward, done, _ = env.step(action)
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        next_states.append(next_state)
        dones.append(done)

        state = next_state
    # 转换为tensor
    states = torch.FloatTensor(states).reshape(-1, 4)
    rewards = torch.FloatTensor(rewards).reshape(-1, 1)
    actions = torch.LongTensor(actions).reshape(-1, 1)       # 动作是0或1
    next_states = torch.FloatTensor(next_states).reshape(-1, 4)
    dones = torch.LongTensor(dones).reshape(-1, 1)

    return states, actions, rewards, next_states, dones

def test():
    state = env.reset()
    reward_sum = 0
    over = False

    while not over:
        action = get_action(state)

        state, reward, over, _ = env.step(action)
        reward_sum += reward

    return reward_sum

# 优势函数
def get_advantage(deltas):
    # 算法来源：GAE，广义优势估计方法。便于计算从后往前累积优势
    advantages = []
    s = 0
    for delta in deltas[::-1]:
        s = 0.98 * 0.95 * s + delta
        advantages.append(s)
    advantages.reverse()

    return advantages

print(get_advantage([0.8, 0.9, 0.99, 1.00, 1.11, 1.12]))

def train():
    optimizer = torch.optim.Adam(actor_model.parameters(), lr=1e-3)
    optimizer_td = torch.optim.Adam(critic_model.parameters(), lr=1e-2)

    # 玩N局游戏，每局游戏玩M次
    for epoch in range(500):
        states, actions, rewards, next_states, dones = get_data()
        # 计算values和targets
        values = critic_model(states)
        targets = critic_model(next_states).detach()    # 目标，不作用梯度
        targets = targets * 0.98
        # 结束状态价值为零
        targets *= (1- dones)
        # 计算总回报(奖励+下一状态)
        targets += rewards

        # 计算价值相对平均的优势，类比策略梯度中的reward_sum
        deltas = (targets - values).squeeze().tolist()  # 标量数值
        advantages = get_advantage(deltas)
        advantages = torch.FloatTensor(advantages).reshape(-1, 1)

        # 取出每一步动作演员给的评分
        old_probs = actor_model(states)
        old_probs = old_probs.gather(dim=1, index=actions)
        old_probs = old_probs.detach()          # 目标，不作用梯度

        # 每批数据反复训练10次
        for _ in range(10):
            # 重新计算每一步动作概率
            new_probs = actor_model(states)
            new_probs = new_probs.gather(dim=1, index=actions)
            # 概率变化率
            ratios = new_probs / old_probs
            # 计算不clip和clip中的loss，取较小值
            no_clip_loss = ratios * advantages
            clip_loss = torch.clamp(ratios, min=0.8, max=1.2) * advantages
            loss = -torch.min(no_clip_loss, clip_loss).mean()
            # 重新计算value，并计算时序差分loss
            values = critic_model(states)
            loss_td = torch.nn.MSELoss()(values, targets)

            # 更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer_td.zero_grad()
            loss_td.backward()
            optimizer_td.step()

        if epoch % 100 == 0:
            result = sum([test() for _ in range(10)]) / 10
            print(epoch, result)

train()
