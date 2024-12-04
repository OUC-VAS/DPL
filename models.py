import torch
import torch.nn as nn
import torch.nn.functional as F
import convnext
from clipiqa.clipiqa_arch import CLIPIQA
import clip
from clipiqa.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from network import PPO, Memory

class DPL(nn.Module):
    def __init__(self, num_classes = 2, quality_genres_num=5, train_stage=1):
        super(DPL, self).__init__()
        self.num_classes = num_classes
        self.quality_genres_num = quality_genres_num
        self.train_stage = train_stage
        self.vqpl = RnnLayer(feature_num=768, hidden_state_dim=768)
        self.fipl = RnnLayer(feature_num=768, hidden_state_dim=768)
        self.backbone = convnext.convnext_tiny(pretrained=True)
        self.vqi = VQI()
        self.fii = FII()
        if train_stage != 1:
            state_dim = 768
            self.fsm = PPO(feature_dim=768, state_dim=state_dim, hidden_state_dim=768, policy_conv=False)
        else:
            self.fsm = None
        self.memory = Memory()
        self.default_mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.default_std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.proj_vqpl = nn.Conv2d(768, 768, kernel_size=1)
        self.proj_fipl = nn.Conv2d(768, 768, kernel_size=1)
        # classifier
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x, training=True, epoch=0):
        list_k1 = self.vqi(x.detach())
        list_k2 = self.fii(x.detach())
        # normalize
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        fea = self.backbone(x)
        batchsize = fea.size(0)
        state = fea
        front_output_vqpl = []
        front_output_fipl = []
        output_vqpl_list = []
        output_fipl_list = []
        for t in range(self.quality_genres_num):
            if t == 0:
                output_vqpl = self.vqpl(self._norm_fea(fea), restart=True, front_output=front_output_vqpl, training=training, train_stage=self.train_stage)
                output_fipl = self.fipl(self._norm_fea(fea), restart=True, front_output=front_output_fipl, training=training, train_stage=self.train_stage)
            else:
                if self.train_stage == 1:
                    action = torch.rand(batchsize, 1).cuda()
                else:
                    if t == 1:
                        action = self.fsm.select_action(self._norm_fea(state), self.memory, restart_batch=True, training=training)
                    else:
                        action = self.fsm.select_action(self._norm_fea(state), self.memory, training=training)
                state = get_patch(state, self._norm_fea(state), action, 256)
                output_vqpl = self.vqpl(self._norm_fea(self.proj_vqpl(state)), restart=False, front_output=front_output_vqpl, training=training, train_stage=self.train_stage)
                output_fipl = self.fipl(self._norm_fea(self.proj_fipl(state)), restart=False, front_output=front_output_fipl, training=training, train_stage=self.train_stage)
            output_vqpl_list.append(output_vqpl)
            output_fipl_list.append(output_fipl)
        # compute fVQ and fFI
        features_vqpl_branch = torch.stack(output_vqpl_list, dim=0)
        features_fipl_branch = torch.stack(output_fipl_list, dim=0)
        f_VQ = torch.gather(features_vqpl_branch, 0, list_k1.unsqueeze(0).unsqueeze(-1).expand(-1, -1, features_vqpl_branch.shape[-1]))
        f_FI = torch.gather(features_fipl_branch, 0, list_k2.unsqueeze(0).unsqueeze(-1).expand(-1, -1, features_fipl_branch.shape[-1]))
        # fuse
        fused_feature = (f_VQ + f_FI) / 2
        pred = self.fc(fused_feature.squeeze(dim=0))
        if training is False:
            return pred
        pred_list = []
        for t in range(self.quality_genres_num):
            if epoch % 2 == 0:
                pred_list.append(self.fc((output_vqpl_list[t] + f_FI.squeeze(dim=0)) / 2.0))
            else:
                pred_list.append(self.fc((output_fipl_list[t] + f_VQ.squeeze(dim=0)) / 2.0))
        if epoch % 2 == 0:
            return pred, pred_list, list_k1
        else:
            return pred, pred_list, list_k2

    
    def _norm_fea(self, fea):
        f = F.adaptive_avg_pool2d(fea, (1,1))
        f = f.view(f.size(0), -1)
        return f


#GRU(refer to https://github.com/blackfeather-wang/GFNet-Pytorch/blob/master/network.py#L186)
class RnnLayer(nn.Module):
    """
        GRU
    """
    def __init__(self, feature_num, hidden_state_dim=1024):
        super(RnnLayer, self).__init__()
        self.feature_num = feature_num

        self.hidden_state_dim = hidden_state_dim

        self.rnn = nn.GRU(feature_num, self.hidden_state_dim)
        self.hidden = None

    def forward(self, x, restart=True, front_output=[], training=True, train_stage=1):
        if restart:
            output, h_n = self.rnn(x.view(1, x.size(0), x.size(1)), torch.zeros(1, x.size(0), self.hidden_state_dim).cuda())
            self.hidden = h_n
        else:
            output, h_n = self.rnn(x.view(1, x.size(0), x.size(1)), self.hidden)
            self.hidden = h_n
        front_output.append(output[0])
        return torch.mean(torch.stack(front_output, dim=0), dim=0)
    
class VQI(nn.Module):
    def __init__(self) -> None:
        super(VQI, self).__init__()
        self.net = CLIPIQA(model_type='clipiqa+').cuda()
        self.net.eval()
        self.intervals = [
            (55, 100),
            (50, 55),
            (45, 50),
            (40, 45),
            (0, 40)
        ]
    
    def map_to_quality_label(self, pred):
        labels = torch.zeros_like(pred, dtype=torch.long)
        for i, interval in enumerate(self.intervals):
            mask = (pred >= interval[0]) & (pred < interval[1])
            labels[mask] = i
        return labels
    
    def forward(self, x):
        with torch.no_grad():
            # Quality prediction
            output = self.net(x)
            pred = output.squeeze(dim=-1) * 100
            pred = torch.round(pred).to(torch.int)
            quality_label = self.map_to_quality_label(pred)
            return quality_label
        

class FII(nn.Module):
    def __init__(self) -> None:
        super(FII, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load("ViT-B/32", device=device, download_root='./pretrained/clip')
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        
        self.text = clip.tokenize(["genuine face", "manipulated face"]).to(device)

        self.intervals = [
            (0, 50),
            (50, 65),
            (65, 80),
            (80, 90),
            (90, 100)
        ]

        self.default_mean = torch.Tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1)
    
    def map_to_quality_label(self, pred):
        labels = torch.zeros_like(pred, dtype=torch.long)
        for i, interval in enumerate(self.intervals):
            if i != len(self.intervals) - 1:
                mask = (pred >= interval[0]) & (pred < interval[1])
            else:
                mask = (pred >= interval[0]) & (pred <= interval[1])
            labels[mask] = i

        return labels

    def forward(self, x):
        with torch.no_grad():
            # preprocess image
            image = (x - self.default_mean.to(x)) / self.default_std.to(x)
            logits_per_image, _ = self.model(image, self.text)
            probs = logits_per_image.softmax(dim=-1)[..., 0] * 100
            pred = torch.round(probs).to(torch.int)
            quality_label = self.map_to_quality_label(pred)
            return quality_label


# utils
def get_patch(x, x_avg, action_sequence, patch_size):
    batch_size = x.size(0)
    channel_num = x.size(1)
    patch_coordinate = torch.floor(action_sequence * (channel_num - patch_size)).int()
    mask = torch.ones(batch_size, channel_num).cuda()
    for i in range(batch_size):
        mask[i, (patch_coordinate[i, 0].item()):((patch_coordinate[i, 0] + patch_size).item())] = 0
    return x - (x_avg * mask).view(batch_size, channel_num, 1, 1)