import torch
import torch.nn as nn
import math
import torch.nn.functional as F
"""refer to https://github.com/blackfeather-wang/GFNet-Pytorch"""

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.hidden = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.hidden[:]

class ActorCritic(nn.Module):
    def __init__(self, feature_dim, state_dim, hidden_state_dim=1024, policy_conv=True, action_std=0.1):
        super(ActorCritic, self).__init__()
        
        # encoder with convolution layer for MobileNetV3, EfficientNet and RegNet
        if policy_conv:
            self.state_encoder = nn.Sequential(
                nn.Conv2d(feature_dim, 32, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(int(state_dim * 32 / feature_dim), hidden_state_dim),
                nn.ReLU()
            )
        # encoder with linear layer for ResNet and DenseNet
        else:
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, 2048),
                nn.ReLU(),
                nn.Linear(2048, hidden_state_dim),
                nn.ReLU()
            )

        self.gru = nn.GRU(hidden_state_dim, hidden_state_dim, batch_first=False)
        self.actor = nn.Sequential(
            nn.Linear(hidden_state_dim, 1),
            nn.Sigmoid())

        self.critic = nn.Sequential(
            nn.Linear(hidden_state_dim, 1))
        
        # self.action_std = torch.full((1,), action_std).cuda()
        self.action_std = nn.Parameter(torch.full((1,), action_std), requires_grad=True)

        self.hidden_state_dim = hidden_state_dim
        self.policy_conv = policy_conv
        self.feature_dim = feature_dim
        self.feature_ratio = int(math.sqrt(state_dim/feature_dim))

    def forward(self):
        raise NotImplementedError

    def act(self, state_ini, memory, restart_batch=False, training=False):
        if restart_batch:
            del memory.hidden[:]
            memory.hidden.append(torch.zeros(1, state_ini.size(0), self.hidden_state_dim).cuda())

        if not self.policy_conv:
            state = state_ini.flatten(1)
        else:
            state = state_ini

        state = self.state_encoder(state)

        state, hidden_output = self.gru(state.view(1, state.size(0), state.size(1)), memory.hidden[-1])
        memory.hidden.append(hidden_output)

        state = state[0]
        action_mean = self.actor(state)
        
        dist = torch.distributions.normal.Normal(action_mean, self.action_std.exp())
        action = dist.sample().cuda()
        if training:
            action = F.relu(action)
            action = 1 - F.relu(1 - action)
            action_logprob = dist.log_prob(action).cuda()
            memory.states.append(state_ini)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
        else:
            action = action_mean

        return action.detach()

    def evaluate(self, state, action):
        seq_l = state.size(0)
        batch_size = state.size(1)

        if not self.policy_conv:
            state = state.flatten(2)
            state = state.view(seq_l * batch_size, state.size(2))
        else:
            state = state.view(seq_l * batch_size, state.size(2), state.size(3), state.size(4))

        state = self.state_encoder(state)
        state = state.view(seq_l, batch_size, -1)

        state, hidden = self.gru(state, torch.zeros(1, batch_size, state.size(2)).cuda())
        state = state.view(seq_l * batch_size, -1)

        action_mean = self.actor(state)

        dist = torch.distributions.normal.Normal(action_mean, self.action_std.exp())

        action_logprobs = dist.log_prob(action.view(seq_l * batch_size, -1)).cuda()
        dist_entropy = dist.entropy().cuda()
        state_value = self.critic(state)

        return action_logprobs.view(seq_l, batch_size), \
               state_value.view(seq_l, batch_size), \
               dist_entropy.view(seq_l, batch_size)

class PPO:
    """
        Proximal policy optimization
    """
    def __init__(self, feature_dim, state_dim, hidden_state_dim, policy_conv,
                 action_std=0.1, lr=0.0003, betas=(0.9, 0.999), gamma=0.7, K_epochs=1, eps_clip=0.2):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(feature_dim, state_dim, hidden_state_dim, policy_conv, action_std).cuda()

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(feature_dim, state_dim, hidden_state_dim, policy_conv, action_std).cuda()
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss(reduction='none')

    def select_action(self, state, memory, restart_batch=False, training=True):
        return self.policy_old.act(state, memory, restart_batch, training)

    def update(self, memory, max_time_steps):
        rewards = []
        discounted_reward = 0

        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.cat(rewards, 0).cuda()

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.stack(memory.states, 0).cuda().detach()
        old_actions = torch.stack(memory.actions, 0).cuda().detach()
        old_logprobs = torch.stack(memory.logprobs, 0).cuda().detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.view(logprobs.shape).detach())

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            # L_REW
            loss_actor = -torch.min(surr1, surr2)
            # L_SE
            loss_critic = 0.5 * self.MseLoss(state_values, rewards)
            # loss2
            loss = loss_actor + loss_critic - 0.01 * dist_entropy

            loss_total = []
            for i in range(max_time_steps.size(0)):
                if(max_time_steps[i] == 0):
                    continue
                loss_total.append(loss[:max_time_steps[i], i])
            if len(loss_total) == 0:
                return 0, 0, 0, rewards
            loss = torch.cat(loss_total, dim=0).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        return loss.item(), loss_actor.mean().item(), loss_critic.mean().item(), rewards