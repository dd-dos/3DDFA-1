import cv2
import torch
import torchvision
import numpy as np
import torch
import torchmetrics
import torchvision.transforms as T
from .params import params_mean_101, params_std_101
from .face3d import face3d
from .ddfa import AverageMeter

fm = face3d.face_model.FaceModel()

@torch.no_grad()
def log_training_samples(imgs, preds, gts, writer, id, job):
    for idx in range(imgs.shape[0]):
        arr_img = de_normalize(imgs[idx], [127.5,127.5,127.5],[128,128,128]).cpu().numpy()
        # arr_img = arr_img.numpy()*255
        arr_img = arr_img.transpose(1,2,0).astype(np.uint8)
        arr_img = np.ascontiguousarray(arr_img, dtype=np.uint8)

        gt_img = arr_img.copy()

        pred = preds[idx].cpu().numpy().reshape(-1,)
        gt = gts[idx].cpu().numpy().reshape(-1,)

        transformed_pred = pred * params_std_101 + params_mean_101
        transformed_gt = gt * params_std_101 + params_mean_101

        pred_pts = fm.reconstruct_vertex(arr_img, transformed_pred)[fm.bfm.kpt_ind][:,:2]
        gt_pts = fm.reconstruct_vertex(arr_img, transformed_gt)[fm.bfm.kpt_ind][:,:2]

        for _pts in pred_pts:
            _pts = tuple(_pts.astype(np.uint8))
            cv2.circle(arr_img, _pts, 2, (0,255,0), -1, 10)
        
        for _pts in gt_pts:
            _pts = tuple(_pts.astype(np.uint8))
            cv2.circle(gt_img, _pts, 2, (0,255,0), -1, 10)

        comparision = np.concatenate((gt_img, arr_img), axis=1)
        comparision = cv2.cvtColor(comparision, cv2.COLOR_BGR2RGB)

        tensor_img = torch.tensor(comparision.transpose(2,0,1)/255.,
                                  dtype=torch.float32).unsqueeze(0)
        grid = torchvision.utils.make_grid(tensor_img)
        writer.add_image(f"{job}/sample-{idx}", grid, id)     


def de_normalize(tensor_img: torch.Tensor, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]):
    de_std = 1/np.array(std)
    de_mean = -np.array(mean)*de_std
    de_normalizer = T.Normalize(de_mean, de_std)
    _img = de_normalizer(tensor_img)

    return _img

class RandomBullShit:
    def __init__(self):
        self.cosine_similarity_meter = AverageMeter()
        self.explained_variance_meter = AverageMeter()
        self.mean_absolute_error_meter = AverageMeter()
        self.mean_abs_percentage_error_meter = AverageMeter()
        self.mean_squared_error_meter = AverageMeter()
        self.mean_squared_log_error_meter = AverageMeter()
        # self.pearson_meter = AverageMeter()
        # self.r2score_meter = AverageMeter()
        # self.spearman_meter = AverageMeter()
        self.smape_meter = AverageMeter()

        self.cosine_similarity = torchmetrics.CosineSimilarity(reduction = 'mean').cuda()
        self.explained_variance = torchmetrics.ExplainedVariance(multioutput ='uniform_average').cuda()
        self.mean_absolute_error = torchmetrics.MeanAbsoluteError().cuda()
        self.mean_abs_percentage_error = torchmetrics.MeanAbsolutePercentageError().cuda()
        self.mean_squared_error = torchmetrics.MeanSquaredError().cuda()
        self.mean_squared_log_error = torchmetrics.MeanSquaredLogError().cuda()
        # self.pearson = torchmetrics.PearsonCorrcoef().cuda()
        # self.r2score = torchmetrics.R2Score(multioutput.cuda()='uniform_average')
        # self.spearman = torchmetrics.SpearmanCorrcoef().cuda()
        self.smape = torchmetrics.SymmetricMeanAbsolutePercentageError().cuda()

    def update(self, preds, gts):
        self.cosine_similarity_meter.update(
            self.cosine_similarity(preds, gts)
        )
        self.explained_variance_meter.update(
            self.explained_variance(preds, gts)
        )
        self.mean_absolute_error_meter.update(
            self.mean_absolute_error(preds, gts)
        )
        self.mean_abs_percentage_error_meter.update(
            self.mean_abs_percentage_error(preds, gts)
        )
        self.mean_squared_error_meter.update(
            self.mean_squared_error(preds, gts)
        )
        self.mean_squared_log_error_meter.update(
            self.mean_squared_log_error(preds, gts)
        )
        # self.pearson_meter.update(
        #     self.pearson(preds, gts)
        # )
        # self.r2score_meter.update(
        #     self.r2score(preds, gts)
        # )
        # self.spearman_meter.update(
        #     self.spearman(preds, gts)
        # )
        self.smape_meter.update(
            self.smape(preds, gts)
        )

    def go(self, writer, id, job):
        writer.add_scalar(
            f'Cosine-Similarity/{job}',
            self.cosine_similarity_meter.avg,
            id
        )

        writer.add_scalar(
            f'Explained-Variance/{job}',
            self.explained_variance_meter.avg,
            id
        )

        writer.add_scalar(
            f'Mean-Absolute-Error/{job}',
            self.mean_absolute_error_meter.avg,
            id
        )

        writer.add_scalar(
            f'Mean-Absolute-Percentage-Error/{job}',
            self.mean_abs_percentage_error_meter.avg,
            id
        )

        writer.add_scalar(
            f'Mean-Squared-Error/{job}',
            self.mean_squared_error_meter.avg,
            id
        )

        writer.add_scalar(
            f'Mean-Squared-Log-Error/{job}',
            self.mean_squared_log_error_meter.avg,
            id
        )

        # writer.add_scalar(
        #     f'Pearson-Corrcoef/{job}',
        #     self.pearson_meter.avg,
        #     id
        # )

        # writer.add_scalar(
        #     f'R2Score/{job}',
        #     self.r2score_meter.avg,
        #     id
        # )

        # writer.add_scalar(
        #     f'Spearman-Corrcoef/{job}',
        #     self.spearman_meter.avg,
        #     id
        # )

        writer.add_scalar(
            f'Symmetric-Mean-Absolute-Percentage-Error/{job}',
            self.smape_meter.avg,
            id
        )

