# coding: utf-8

__author__ = 'cleardusk'

import sys

sys.path.append('..')

import os.path as osp
import numpy as np
from utils.io import _load

make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)
bfm_path = make_abs_path('bfm_noneck_v3.pkl')
mean_std_path = make_abs_path('param_mean_std_62d_120x120.pkl')

def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr


def parse_param(param):
    """matrix pose form
    param: shape=(trans_dim+shape_dim+exp_dim,), i.e., 62 = 12 + 40 + 10
    """

    # pre-defined templates for parameter
    n = param.shape[0]
    if n == 62:
        trans_dim, shape_dim, exp_dim = 12, 40, 10
    elif n == 72:
        trans_dim, shape_dim, exp_dim = 12, 40, 20
    elif n == 141:
        trans_dim, shape_dim, exp_dim = 12, 100, 29
    else:
        raise Exception(f'Undefined templated param parsing rule')

    R_ = param[:trans_dim].reshape(3, -1)
    R = R_[:, :3]
    offset = R_[:, -1].reshape(3, 1)
    alpha_shp = param[trans_dim:trans_dim + shape_dim].reshape(-1, 1)
    alpha_exp = param[trans_dim + shape_dim:].reshape(-1, 1)

    return R, offset, alpha_shp, alpha_exp


def similar_transform(pts3d, roi_box, size):
    pts3d[0, :] -= 1  # for Python compatibility
    pts3d[2, :] -= 1
    pts3d[1, :] = size - pts3d[1, :]

    sx, sy, ex, ey = roi_box
    scale_x = (ex - sx) / size
    scale_y = (ey - sy) / size
    pts3d[0, :] = pts3d[0, :] * scale_x + sx
    pts3d[1, :] = pts3d[1, :] * scale_y + sy
    s = (scale_x + scale_y) / 2
    pts3d[2, :] *= s
    pts3d[2, :] -= np.min(pts3d[2, :])
    return np.array(pts3d, dtype=np.float32)


class BFMModel(object):
    def __init__(self, bfm_fp=bfm_path, shape_dim=40, exp_dim=10):
        bfm = _load(bfm_fp)
        self.u = bfm.get('u').astype(np.float32)  # fix bug
        self.w_shp = bfm.get('w_shp').astype(np.float32)[..., :shape_dim]
        self.w_exp = bfm.get('w_exp').astype(np.float32)[..., :exp_dim]
        # self.w_shp = bfm.get('w_shp').astype(np.float32)
        # self.w_exp = bfm.get('w_exp').astype(np.float32)
        if osp.split(bfm_fp)[-1] == 'bfm_noneck_v3.pkl':
            self.tri = _load(make_abs_path('tri.pkl'))  # this tri/face is re-built for bfm_noneck_v3
        else:
            self.tri = bfm.get('tri')

        self.tri = _to_ctype(self.tri.T).astype(np.int32)
        self.keypoints = bfm.get('keypoints').astype(np.long)  # fix bug
        w = np.concatenate((self.w_shp, self.w_exp), axis=1)
        self.w_norm = np.linalg.norm(w, axis=0)

        self.u_base = self.u[self.keypoints].reshape(-1, 1)
        self.w_shp_base = self.w_shp[self.keypoints]
        self.w_exp_base = self.w_exp[self.keypoints]
        mean_std = _load(mean_std_path)
        self.param_mean = mean_std['mean']
        self.param_std = mean_std['std']

    def reconstruct_vertex(self, param, dense=False):
        param = param * self.param_std + self.param_mean

        if dense:
            R, offset, alpha_shp, alpha_exp = parse_param(param)
            pts3d = R @ (self.u + self.w_shp @ alpha_shp + self.w_exp @ alpha_exp). \
                reshape(3, -1, order='F') + offset
        else:
            R, offset, alpha_shp, alpha_exp = parse_param(param)
            pts3d = R @ (self.u_base + self.w_shp_base @ alpha_shp + self.w_exp_base @ alpha_exp). \
                reshape(3, -1, order='F') + offset
        
        pts3d[1, :] = 120 + 1 - pts3d[1, :]

        return pts3d
        
