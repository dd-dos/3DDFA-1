#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
from utils.io import _load, _numpy_to_cuda, _numpy_to_tensor
from utils.params import keypoints
from utils.face3d.face3d.face_model import FaceModel
from utils.face3d.utils import show_vertices
fm = FaceModel()

# _to_tensor = _numpy_to_cuda  # gpu
_to_tensor = _numpy_to_tensor

def parse_param_batch(param):
    """Work for both numpy and tensor"""
    N = param.shape[0]
    p_ = param[:, :12].view(N, 3, -1)
    p = p_[:, :, :3]
    offset = p_[:, :, -1].view(N, 3, 1)
    alpha_shp = param[:, 12:72].view(N, -1, 1)
    alpha_exp = param[:, 72:].view(N, -1, 1)
    return p, offset, alpha_shp, alpha_exp


class VDCLoss(nn.Module):
    def __init__(self, opt_style='all', resample_num=0):
        super(VDCLoss, self).__init__()

        # self.u = _to_tensor(u)
        self.param_mean = _to_tensor(fm.bfm.params_mean_101)
        self.param_std = _to_tensor(fm.bfm.params_std_101)
        # self.w_shp = _to_tensor(w_shp)
        # self.w_exp = _to_tensor(w_exp)

        self.u = _to_tensor(fm.bfm.model['shapeMU'])
        self.w_shp = _to_tensor(fm.bfm.model['shapePC'][:,:60])
        self.w_exp = _to_tensor(fm.bfm.model['expPC'][:,:29])

        self.keypoints = _to_tensor(keypoints)
        self.u_base = self.u[self.keypoints]
        self.w_shp_base = self.w_shp[self.keypoints]
        self.w_exp_base = self.w_exp[self.keypoints]

        self.w_shp_length = self.w_shp.shape[0] // 3

        self.opt_style = opt_style
        self.resample_num = resample_num

    def reconstruct_and_parse(self, input, target):
        # reconstruct
        param = input * self.param_std + self.param_mean
        param_gt = target * self.param_std + self.param_mean

        # parse param
        p, offset, alpha_shp, alpha_exp = parse_param_batch(param)
        pg, offsetg, alpha_shpg, alpha_expg = parse_param_batch(param_gt)

        return (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg)

    def forward_all(self, input, target):
        (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg) \
            = self.reconstruct_and_parse(input, target)

        N = input.shape[0]
        offset[:, -1] = offsetg[:, -1]
        gt_vertex = pg @ (self.u + self.w_shp @ alpha_shpg + self.w_exp @ alpha_expg) \
            .view(N, -1, 3).permute(0, 2, 1) + offsetg
        vertex = p @ (self.u + self.w_shp @ alpha_shp + self.w_exp @ alpha_exp) \
            .view(N, -1, 3).permute(0, 2, 1) + offset

        diff = (gt_vertex - vertex) ** 2
        loss = torch.mean(diff)
        return loss

    def forward_resample(self, input, target):
        (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg) \
            = self.reconstruct_and_parse(input, target)

        # resample index
        index = torch.randperm(self.w_shp_length)[:self.resample_num].reshape(-1, 1)
        # keypoints_resample = torch.cat((3 * index, 3 * index + 1, 3 * index + 2), dim=1).view(-1).cuda()
        keypoints_resample = torch.cat((3 * index, 3 * index + 1, 3 * index + 2), dim=1).view(-1)

        keypoints_mix = torch.cat((self.keypoints, keypoints_resample))
        w_shp_base = self.w_shp[keypoints_mix]
        u_base = self.u[keypoints_mix]
        w_exp_base = self.w_exp[keypoints_mix]

        offset[:, -1] = offsetg[:, -1]

        N = input.shape[0]
        # import ipdb; ipdb.set_trace(context=10)
        # with torch.cuda.amp.autocast(enabled=False):
        gt_vertex = pg @ (u_base + w_shp_base @ alpha_shpg + w_exp_base @ alpha_expg) \
            .view(N, -1, 3).permute(0, 2, 1) + offsetg
        vertex = p @ (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp) \
            .view(N, -1, 3).permute(0, 2, 1) + offset

        show_vertices(gt_vertex, '3D')
        import ipdb; ipdb.set_trace(context=10)
        diff = (gt_vertex - vertex) ** 2
        loss = torch.mean(diff)
        return loss

    def forward(self, input, target):
        if self.opt_style == 'all':
            return self.forward_all(input, target)
        elif self.opt_style == 'resample':
            return self.forward_resample(input, target)
        else:
            raise Exception(f'Unknown opt style: f{opt_style}')


if __name__ == '__main__':
    pass
