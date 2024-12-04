import torch
from models import DPL
import torch.nn.functional as F
from focal_loss import focal_loss

class Trainer():
    def __init__(self, gpu_ids, quality_genres_num, train_stage):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
        self.quality_genres_num = quality_genres_num
        self.model = DPL(quality_genres_num=quality_genres_num, train_stage=train_stage).to(self.device)
        self.train_stage = train_stage
        if train_stage == 2:
            for name, param in self.model.named_parameters():
                param.requires_grad = False
        # print parameter
        for name, param in self.model.named_parameters():
            print(name, param.requires_grad)
        if self.model.fsm is not None:
            for name, param in self.model.fsm.policy.named_parameters():
                print(name, param.requires_grad)
        
        self.optimizer = torch.optim.Adam([
            {'params': list(filter(lambda p: p.requires_grad, self.model.parameters())), 'lr': 5e-5},
        ])
        self.criterion_fn = focal_loss(num_classes=2, reduction='mean')
    
    def set_input(self, input_list, label):
        self.input_list = []
        self.jpeg_quality_list = []
        for i in range(len(input_list)):
            self.input_list.append(input_list[i].to(self.device))
        self.label = label.to(self.device)
    
    def forward(self, x):
        pred = self.model(x, training=False)
        return pred
    
    def optimize_weight(self, reward_list, epoch):
        input = torch.cat(self.input_list, dim=0)
        pred, pred_list, list_k = self.model(input, epoch = epoch)
        confidence_last = 0
        for t in range(self.quality_genres_num):
            confidence = torch.gather(F.softmax(pred_list[t].detach(), 1), dim=1, index=self.label.view(-1, 1)).view(1, -1)
            reward = confidence - confidence_last
            confidence_last = confidence
            if t != 0:
                self.model.memory.rewards.append(reward)
                reward_list[t-1].update(reward.data.mean().item(), input.size(0))
        
        if self.train_stage == 1:
            loss_cls = self.criterion_fn(pred, self.label)
            # compute samples' hardness
            hardness = F.cross_entropy(pred.detach(), self.label, reduction='none') - 0.6931
            hardness_mask = (hardness > 0.0)
            len_ = sum(hardness_mask).item()
            entropy_items = torch.distributions.categorical.Categorical(F.softmax(pred, dim=-1)).entropy() * hardness_mask
            loss_entropy = - entropy_items.sum() / max(len_, 1)
            loss1 = loss_cls + loss_entropy
            self.optimizer.zero_grad()
            loss1.backward()
            self.optimizer.step()
        elif self.train_stage == 2:
            loss2, loss_actor, loss_critic, reward_w = self.model.fsm.update(self.model.memory, list_k)
        self.model.memory.clear_memory()
        if self.train_stage == 1:
            return loss1.item(), loss_cls.item(), loss_entropy.item()
        elif self.train_stage == 2:
            _reward = [reward.ave for reward in reward_list]
            return loss2, loss_actor, loss_critic, _reward, reward_w.mean(dim=1).tolist()