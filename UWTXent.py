import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class UWTXent(nn.Module):

    def __init__(self, t=8, eps=1e-12, dim=256, divide_num=1.):
        super(UWTXent, self).__init__()
        self.t = t
        self.eps = eps
        self.dim = divide_num * dim
        self.add = 1.1 / divide_num
        self.divide_num = divide_num

    def forward(self, out_1, out_2):
        # l2_out_1 = torch.sqrt(torch.sum(out_1 ** 2, dim=1)+self.eps)
        # l2_out_2 = torch.sqrt(torch.sum(out_2 ** 2, dim=1)+self.eps)
        # sim = torch.pow((torch.sum(out_1 * out_2, dim=1)/l2_out_1 / l2_out_2 / self.divide_num + self.add), self.t)+self.eps

        sim = torch.pow((torch.sum(out_1 * out_2, dim=1)/self.dim + self.add), self.t)+self.eps
        # 2b
        sim = torch.cat([sim, sim], dim=0)
        # l2_out = torch.unsqueeze(torch.cat([l2_out_1, l2_out_2], dim=0), dim=0)

        # 2b * fd
        sim_matrix = torch.cat([out_1, out_2], dim=0)
        # 2b * 2b
        # sim_matrix = torch.pow(torch.mm(sim_matrix, torch.t(sim_matrix)) / torch.mm(torch.t(l2_out), l2_out) / self.divide_num + self.add, self.t)+self.eps
        sim_matrix = torch.pow(torch.mm(sim_matrix, torch.t(sim_matrix)) / self.dim + self.add, self.t)+self.eps
        mask_matrix = (torch.ones_like(sim_matrix) - torch.eye(sim_matrix.shape[0], device=sim_matrix.device)).bool()
        # 2b * (2b - 1)
        sim_matrix = torch.masked_select(sim_matrix, mask_matrix).view(sim_matrix.shape[0], -1)
        # loss
        loss = torch.mean(-torch.log(sim / torch.sum(sim_matrix, dim=1)))
        return loss