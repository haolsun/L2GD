import copy
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.attention import MultiHeadAttn, SelfAttn


class Classifier(nn.Module):
    def __init__(self, base_model, num_classes, n_features, with_softmax=True):
        super(Classifier, self).__init__()
        self.base_model = base_model

        self.fc = nn.Linear(n_features, num_classes)
        self.fc.bias.data.zero_()

        self.with_softmax = with_softmax

    def forward(self, x):
        out = self.base_model(x)
        out = self.fc(out)  # [B,K]

        if self.with_softmax:
            out = F.softmax(out, dim=-1)
        return out


class ClassifierRejector(nn.Module):
    def __init__(self, base_model, num_classes, n_features, n_experts=1, with_softmax=True, decouple=False):
        super(ClassifierRejector, self).__init__()
        self.base_model_clf = base_model
        base_mdl_lst = [self.base_model_clf]
        if decouple:
            self.base_model_rej = copy.deepcopy(self.base_model_clf)
            base_mdl_lst += [self.base_model_rej]
        else:
            self.base_model_rej = self.base_model_clf

        self.fc_clf = nn.Linear(n_features, num_classes)
        self.fc_clf.bias.data.zero_()

        self.fc_rej = nn.Linear(n_features, n_experts)
        self.fc_rej.bias.data.zero_()

        self.with_softmax = with_softmax
        self.params = nn.ModuleDict({
            'base': nn.ModuleList(base_mdl_lst),
            'clf': nn.ModuleList([self.fc_clf, self.fc_rej])
        })

    def forward(self, x):
        out = self.base_model_clf(x)
        logits_clf = self.fc_clf(out)  # [B,K]

        out = self.base_model_rej(x)
        logit_rej = self.fc_rej(out)  # [B,1]

        out = torch.cat([logits_clf, logit_rej], -1)  # [B,K+1]

        if self.with_softmax:
            out = F.softmax(out, dim=-1)
        return out


def get_activation(act_str):
    if act_str == 'relu':
        return functools.partial(nn.ReLU, inplace=True)
    elif act_str == 'elu':
        return functools.partial(nn.ELU, inplace=True)
    else:
        raise ValueError('invalid activation')


def build_mlp(dim_in, dim_hid, dim_out, depth, activation='relu'):
    act = get_activation(activation)
    if depth == 1:
        modules = [nn.Linear(dim_in, dim_out)]  # no hidden layers
    else:  # depth>1
        modules = [nn.Linear(dim_in, dim_hid), act()]
        for _ in range(depth - 2):
            modules.append(nn.Linear(dim_hid, dim_hid))
            modules.append(act())
        modules.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*modules)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ExpertWiseAggregator(nn.Module):
    def __init__(self, base_model, num_classes, n_features, dim_hid=128, depth_embed=6, depth_rej=4,
                 dim_class_embed=128,
                 with_attn=False, with_softmax=True, decouple=False):
        super(ExpertWiseAggregator, self).__init__()
        self.num_classes = num_classes
        self.with_attn = with_attn
        self.with_softmax = with_softmax
        self.decouple = decouple
        self.dim_hid = dim_hid
        self.n_features = n_features

        self.base_model = base_model
        base_mdl_lst = [self.base_model]
        if self.decouple:
            self.base_model_rej = copy.deepcopy(self.base_model)
            base_mdl_lst += [self.base_model_rej]
        else:
            self.base_model_rej = self.base_model

        self.fc = nn.Linear(n_features, num_classes)
        self.fc.bias.data.zero_()

        self.embed_class = nn.Embedding(num_classes, dim_class_embed)
        self.rejector = build_mlp(n_features + dim_hid, dim_hid, 1, depth_rej)
        self.rejector[-1].bias.data.zero_()

        if not self.with_attn:
            self.embed = build_mlp(n_features + dim_class_embed * 2, dim_hid, dim_hid, depth_embed)
        else:
            self.embed = nn.Sequential(
                build_mlp(n_features + dim_class_embed * 2, dim_hid, dim_hid, depth_embed - 2),
                nn.ReLU(True),
                SelfAttn(dim_hid, dim_hid)
            )

        # self.embed_post = build_mlp_fixup(dim_hid, dim_hid, dim_hid, 2)
        rej_mdl_lst = [self.rejector, self.embed_class, self.embed]  # self.embed_post

        if with_attn:
            self.attn = MultiHeadAttn(n_features, n_features, dim_hid, dim_hid)
            rej_mdl_lst += [self.attn]

        self.params = nn.ModuleDict({
            'base': nn.ModuleList(base_mdl_lst),
            'clf': nn.ModuleList([self.fc]),
            'rej': nn.ModuleList(rej_mdl_lst)
        })

    def forward(self, x, cntxt=None):
        '''
        Args:
            x : tensor [B,3,32,32]
            cntxt : AttrDict, with entries
                xc : tensor [E,Nc,3,32,32]
                yc : tensor [E,Nc]
                mc : tensor [E,Nc]
        '''
        if cntxt is None:
            n_experts = 1
        else:
            n_experts = cntxt.xc.shape[0] # meta_batch_size [E]=4

        # 1.把batch图片传入base，经过fc，得到10分类的概率
        x_embed = self.base_model(x)  # x:[B,3,32,32] -> x_embed:[B,Dx]
        logits_clf = self.fc(x_embed)  # x_embed:[B,Dx] -> logits_clf:[B,K]
        # logits_clf = logits_clf.unsqueeze(0).repeat(n_experts, 1, 1) #logits_clf:[B,K] -> logits_clf:[E,B,K]

        # 2. 把batch图片和cntx图片混合嵌入
        if cntxt is None:
            embedding = torch.zeros((n_experts, x.shape[0], self.dim_hid), device=x_embed.device)
        else:
            embedding = self.encode(cntxt, x)  # [E,B,H]

        x_embed = self.base_model_rej(x)  # [B,Dx]
        x_embed = x_embed.unsqueeze(0).repeat(n_experts, 1, 1)  # [E,B,Dx]

        packed = torch.cat([x_embed, embedding], -1)  # [E,B,Dx+H]

        logit_rej = self.rejector(packed)  # [E,B,1]
        logit_rej = logit_rej.permute(1,0,2).reshape(x.shape[0],n_experts)

        out = torch.cat([logits_clf, logit_rej], -1)  # [B,K+E]
        if self.with_softmax:
            out = F.softmax(out, dim=-1)
        return out

    def encode(self, cntxt, xt):

        n_experts = cntxt.xc.shape[0]
        batch_size = xt.shape[0]

        cntxt_xc = cntxt.xc.view((-1,) + cntxt.xc.shape[-3:])  # [E*Nc,3,32,32]
        if not self.decouple:
            with torch.no_grad():
                xc_embed = self.base_model_rej(cntxt_xc)  # [E*Nc,Dx]
            xc_embed = xc_embed.detach()
        else:
            xc_embed = self.base_model_rej(cntxt_xc)  # [E*Nc,Dx]
        xc_embed = xc_embed.view(cntxt.xc.shape[:2] + (xc_embed.shape[-1],))  # E,Nc,Dx]

        # 获取所有bz，y标签嵌入和每一个专家的嵌入
        yc_embed = self.embed_class(cntxt.yc)  # [E,Nc,H]
        mc_embed = self.embed_class(cntxt.mc)  # [E,Nc,H]
        out = torch.cat([xc_embed, yc_embed, mc_embed], -1)  # [E,Nc,Dx+2H]

        out = self.embed(out)  # [E,Nc,H]

        if not self.with_attn:
            embedding = out.mean(-2)  # [E, H]
            embedding = embedding.unsqueeze(1).repeat(1,batch_size, 1)  # [E,B,H]
        else:
            xt_embed = self.base_model_rej(xt)  # [B,Dx]
            if not self.decouple:
                xt_embed = xt_embed.detach()  # stop gradients flowing
            xt_embed = xt_embed.unsqueeze(0).repeat(n_experts, 1, 1)  # [E,B,Dx]
            embedding = self.attn(xt_embed, xc_embed, out)  # [E,B,H]

        return embedding



class ClassifierRejectorWithContextEmbedder(nn.Module):
    def __init__(self, base_model, num_classes, n_features, n_experts=1, dim_hid=128, depth_embed=6, depth_rej=4,
                 dim_class_embed=128,
                 with_attn=False, with_softmax=True, decouple=False):
        super(ClassifierRejectorWithContextEmbedder, self).__init__()
        self.num_classes = num_classes
        self.with_attn = with_attn
        self.with_softmax = with_softmax
        self.decouple = decouple
        self.dim_hid = dim_hid
        self.n_features = n_features

        self.base_model = base_model
        base_mdl_lst = [self.base_model]
        if self.decouple:
            self.base_model_rej = copy.deepcopy(self.base_model)
            base_mdl_lst += [self.base_model_rej]
        else:
            self.base_model_rej = self.base_model

        self.fc = nn.Linear(n_features, num_classes)
        self.fc.bias.data.zero_()

        self.embed_class = nn.Embedding(num_classes, dim_class_embed)
        self.rejector = build_mlp(n_features + dim_hid, dim_hid, n_experts, depth_rej)
        self.rejector[-1].bias.data.zero_()

        if not self.with_attn:
            self.embed = build_mlp(n_features + dim_class_embed * (1+n_experts), dim_hid, dim_hid, depth_embed)
        else:
            self.embed = nn.Sequential(
                build_mlp(n_features + dim_class_embed * (1+n_experts), dim_hid, dim_hid, depth_embed - 2),
                nn.ReLU(True),
                SelfAttn(dim_hid, dim_hid)
            )

        rej_mdl_lst = [self.rejector, self.embed_class, self.embed]  # self.embed_post

        if with_attn:
            self.attn = MultiHeadAttn(n_features, n_features, dim_hid, dim_hid)
            rej_mdl_lst += [self.attn]

        self.params = nn.ModuleDict({
            'base': nn.ModuleList(base_mdl_lst),
            'clf': nn.ModuleList([self.fc]),
            'rej': nn.ModuleList(rej_mdl_lst)
        })

    def forward(self, x, cntxt=None):
        '''
        Args:
            x : tensor [B,3,32,32]
            cntxt : AttrDict, with entries
                xc : tensor [E,Nc,3,32,32]
                yc : tensor [E,Nc]
                mc : tensor [E,Nc]
        '''
        if cntxt is None:
            n_experts = 1
        else:
            n_experts = cntxt.xc.shape[0] #meta_batch_size


        x_embed = self.base_model(x)  # x:[B,3,32,32] -> x_embed:[B,Dx]
        logits_clf = self.fc(x_embed)  # x_embed:[B,Dx] -> logits_clf:[B,K]
        logits_clf = logits_clf.unsqueeze(0).repeat(n_experts, 1, 1) # logits_clf:[B,K] -> logits_clf:[E,B,K]


        if cntxt is None:
            embedding = torch.zeros((n_experts, x.shape[0], self.dim_hid), device=x_embed.device)
        else:
            embedding = self.encode(cntxt, x)  # [MBZ,B,H]

        x_embed = self.base_model_rej(x)  # [B,Dx]
        x_embed = x_embed.unsqueeze(0).repeat(n_experts, 1, 1)  # [MBZ,B,Dx]

        packed = torch.cat([x_embed, embedding], -1)  # [MBZ,B,Dx+H]

        logit_rej = self.rejector(packed)  # [MBZ,B,n_exp]

        out = torch.cat([logits_clf, logit_rej], -1)  # [MBZ, B,K+n_exp]
        if self.with_softmax:
            out = F.softmax(out, dim=-1)
        return out

    def encode(self, cntxt, xt):
        MBZ = len(cntxt.mc)
        n_experts = cntxt.mc[0].shape[0]
        batch_size = xt.shape[0]

        cntxt_xc = cntxt.xc.view((-1,) + cntxt.xc.shape[-3:])  # [MBZ*Nc,3,32,32]
        if not self.decouple:
            with torch.no_grad():
                xc_embed = self.base_model_rej(cntxt_xc)  # [MBZ*Nc,Dx]
            xc_embed = xc_embed.detach()
        else:
            xc_embed = self.base_model_rej(cntxt_xc)  # [MBZ*Nc,Dx]

        xc_embed = xc_embed.view(cntxt.xc.shape[:2] + (xc_embed.shape[-1],))  # [MBZ,Nc,Dx]

        yc_embed = self.embed_class(cntxt.yc)  # [MBZ,Nc,H]
        mc_embed = []
        for i in range(MBZ):
            for j in range(n_experts):
                mc_embed_ij = self.embed_class(cntxt.mc[i][j])  # [Nc,H] -> [Nc, n*H]
                if j == 0:
                    mc_embed_tmp = mc_embed_ij
                else:
                    mc_embed_tmp = torch.cat([mc_embed_tmp, mc_embed_ij],dim=-1)
            mc_embed.append(mc_embed_tmp.unsqueeze(0)) #list:bz [1,Nc,n*H]
        mc_embed = torch.vstack(mc_embed)# [MBZ, Nc, n*H]
        out = torch.cat([xc_embed, yc_embed, mc_embed], -1)  # [MBZ,Nc,Dx+4H]

        out = self.embed(out)  # [MBZ,Nc,H]


        if not self.with_attn:
            embedding = out.mean(-2)  # [MBZ, H]
            embedding = embedding.unsqueeze(1).repeat(1,batch_size, 1)  # [MBZ,B,H]
        else:
            xt_embed = self.base_model_rej(xt)  # [B,Dx]
            if not self.decouple:
                xt_embed = xt_embed.detach()  # stop gradients flowing
            xt_embed = xt_embed.unsqueeze(0).repeat(MBZ, 1, 1)  # [MBZ,B,Dx]
            embedding = self.attn(xt_embed, xc_embed, out)  # [MBZ,B,H]

        return embedding

