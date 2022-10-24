from typing import List, Optional, Tuple, Union

import torch
import numpy as np
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale, is_norm
from mmengine.model import bias_init_with_prob, constant_init, normal_init
from mmengine.structures import InstanceData
from torch import Tensor
from mmcv.ops import DeformConv2d
import torch.nn.functional as F

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import distance2bbox
from mmdet.utils import reduce_mean
from mmdet.utils.typing import ConfigType, InstanceList, OptInstanceList, RangeType, OptConfigType
from ..layers.transformer import inverse_sigmoid
from ..task_modules import anchor_inside_flags
from ..utils import (images_to_levels, multi_apply, sigmoid_geometric_mean,
                     unmap)
from .atss_head import ATSSHead

INF = 1e8

class TaskDecomposition(nn.Module):
    """Task decomposition module in task-aligned predictor of TOOD.

    Args:
        feat_channels (int): Number of feature channels in TOOD head.
        stacked_convs (int): Number of conv layers in TOOD head.
        la_down_rate (int): Downsample rate of layer attention.
            Defaults to 8.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional):  Config dict for
        normalization layer. Defaults to None.
    """

    def __init__(self,
                 feat_channels: int,
                 stacked_convs: int,
                 la_down_rate: int = 8,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None) -> None:
        super().__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs
        self.norm_cfg = norm_cfg
        self.layer_attention = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.in_channels // la_down_rate,
                self.stacked_convs,
                1,
                padding=0), nn.Sigmoid())

        self.reduction_conv = ConvModule(
            self.in_channels,
            self.feat_channels,
            1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=norm_cfg is None)

    def init_weights(self) -> None:
        """Initialize the parameters."""
        for m in self.layer_attention.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        normal_init(self.reduction_conv.conv, std=0.01)

    def forward(self,
                feat: Tensor,
                avg_feat: Optional[Tensor] = None) -> Tensor:
        """Forward function of task decomposition module."""
        b, c, h, w = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        weight = self.layer_attention(avg_feat)

        # here we first compute the product between layer attention weight and
        # conv weight, and then compute the convolution between new conv weight
        # and feature map, in order to save memory and FLOPs.
        conv_weight = weight.reshape(
            b, 1, self.stacked_convs,
            1) * self.reduction_conv.conv.weight.reshape(
                1, self.feat_channels, self.stacked_convs, self.feat_channels)
        conv_weight = conv_weight.reshape(b, self.feat_channels,
                                          self.in_channels)
        feat = feat.reshape(b, self.in_channels, h * w)
        feat = torch.bmm(conv_weight, feat).reshape(b, self.feat_channels, h,
                                                    w)
        if self.norm_cfg is not None:
            feat = self.reduction_conv.norm(feat)
        feat = self.reduction_conv.activate(feat)

        return feat




@MODELS.register_module()
class ZFHead7(ATSSHead):
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 with_objectness: bool = True,
                 act_cfg: ConfigType = dict(type='ReLU'),
                 **kwargs) -> None:
        self.act_cfg = act_cfg
        self.with_objectness = with_objectness
        self.bbox_norm_type: str = 'reg_denom'
        self.gradient_mul: float = 0.1
        super().__init__(num_classes, in_channels, **kwargs)
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg.assigner)
        


    def _init_layers(self) -> None:
        # self.atss = ATSSHead(num_classes = self.num_classes, in_channels = self.in_channels)
        self.num_dconv_points = 9
        self.dcn_kernel = int(np.sqrt(self.num_dconv_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)                             

        self.relu = nn.ReLU()
        self.vfnet_reg_conv = ConvModule(
            self.feat_channels, 
            self.feat_channels, 
            3, 
            stride=1, 
            padding=1, 
            conv_cfg=self.conv_cfg, 
            norm_cfg=self.norm_cfg)
        self.vfnet_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.prior_generator.strides])

        self.vfnet_reg_refine_dconv = DeformConv2d(
            self.feat_channels, 
            self.feat_channels, self.dcn_kernel, 
            1, 
            padding=self.dcn_pad)
        self.vfnet_reg_refine = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.scales_refine = nn.ModuleList([Scale(1.0) for _ in self.prior_generator.strides])

        self.vfnet_cls_dconv = DeformConv2d(
            self.feat_channels,
            self.feat_channels,
            self.dcn_kernel,
            1,
            padding=self.dcn_pad)
        self.vfnet_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        
        self.cls_reg_convs = nn.ModuleList()
        for i in range(4):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg= dict(type='GN', num_groups=32, requires_grad=True)))
        self.cls_decomp = TaskDecomposition(feat_channels=256,
                                            stacked_convs=4,
                                            la_down_rate=32,
                                            conv_cfg=self.conv_cfg,
                                            norm_cfg= dict(type='GN', num_groups=32, requires_grad=True))
        self.reg_decomp = TaskDecomposition(feat_channels=256,
                                            stacked_convs=4,
                                            la_down_rate=32,
                                            conv_cfg=self.conv_cfg,
                                            norm_cfg= dict(type='GN', num_groups=32, requires_grad=True))    
    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)
        

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:

            - cls_scores (list[Tensor]): Box iou-aware scores for each scale
              level, each is a 4D-tensor, the channel number is
              num_points * num_classes.
            - bbox_preds (list[Tensor]): Box offsets for each
              scale level, each is a 4D-tensor, the channel number is
              num_points * 4.
            - bbox_preds_refine (list[Tensor]): Refined Box offsets for
              each scale level, each is a 4D-tensor, the channel
              number is num_points * 4.
        """
        regress_ranges: RangeType = ((-1, 64), (64, 128), (128, 256))
        reg_denoms = [
            regress_range[-1] for regress_range in regress_ranges
        ]
        reg_denoms[-1] = reg_denoms[-2] * 2
        strides = [8, 16, 32]

        return multi_apply(self.forward_single, x, self.scales,
                           self.scales_refine, strides, reg_denoms)

    def forward_single(self, x: Tensor, scale: Scale, scale_refine: Scale,
                       stride: int, reg_denom: int) -> tuple:
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            scale_refine (:obj: `mmcv.cnn.Scale`): Learnable scale module to
                resize the refined bbox prediction.
            stride (int): The corresponding stride for feature maps,
                used to normalize the bbox prediction when
                bbox_norm_type = 'stride'.
            reg_denom (int): The corresponding regression range for feature
                maps, only used to normalize the bbox prediction when
                bbox_norm_type = 'reg_denom'.

        Returns:
            tuple: iou-aware cls scores for each box, bbox predictions and
            refined bbox predictions of input feature maps.
        """

        cls_reg_feat = []
        for cls_reg_layer in self.cls_reg_convs:
            x = cls_reg_layer(x)
            cls_reg_feat.append(x)
        feat = torch.cat(cls_reg_feat, 1)
        avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))


        cls_feat = self.reg_decomp(feat, avg_feat)
        reg_feat = self.cls_decomp(feat, avg_feat)

        reg_feat_init = self.vfnet_reg_conv(reg_feat)
        if self.bbox_norm_type == 'reg_denom':
            bbox_pred = scale(
                self.vfnet_reg(reg_feat_init)).float().exp() * reg_denom
        elif self.bbox_norm_type == 'stride':
            bbox_pred = scale(
                self.vfnet_reg(reg_feat_init)).float().exp() * stride
        else:
            raise NotImplementedError

        # compute star deformable convolution offsets
        # converting dcn_offset to reg_feat.dtype thus VFNet can be
        # trained with FP16
        dcn_offset = self.star_dcn_offset(bbox_pred, self.gradient_mul,
                                          stride).to(reg_feat.dtype)

        # refine the bbox_pred
        reg_feat = self.relu(self.vfnet_reg_refine_dconv(reg_feat, dcn_offset))
        bbox_pred_refine = scale_refine(
            self.vfnet_reg_refine(reg_feat)).float().exp()
        bbox_pred_refine = bbox_pred_refine * bbox_pred.detach()

        # predict the iou-aware cls score
        cls_feat = self.relu(self.vfnet_cls_dconv(cls_feat, dcn_offset))
        cls_score = self.vfnet_cls(cls_feat)

        if self.training:
            return cls_score, bbox_pred, bbox_pred_refine
        else:
            return cls_score, bbox_pred_refine

    def star_dcn_offset(self, bbox_pred: Tensor, gradient_mul: float,
                        stride: int) -> Tensor:
        """Compute the star deformable conv offsets.

        Args:
            bbox_pred (Tensor): Predicted bbox distance offsets (l, r, t, b).
            gradient_mul (float): Gradient multiplier.
            stride (int): The corresponding stride for feature maps,
                used to project the bbox onto the feature map.

        Returns:
            Tensor: The offsets for deformable convolution.
        """
        dcn_base_offset = self.dcn_base_offset.type_as(bbox_pred)
        bbox_pred_grad_mul = (1 - gradient_mul) * bbox_pred.detach() + \
            gradient_mul * bbox_pred
        # map to the feature map scale
        bbox_pred_grad_mul = bbox_pred_grad_mul / stride
        N, C, H, W = bbox_pred.size()

        x1 = bbox_pred_grad_mul[:, 0, :, :]
        y1 = bbox_pred_grad_mul[:, 1, :, :]
        x2 = bbox_pred_grad_mul[:, 2, :, :]
        y2 = bbox_pred_grad_mul[:, 3, :, :]
        bbox_pred_grad_mul_offset = bbox_pred.new_zeros(
            N, 2 * self.num_dconv_points, H, W)
        bbox_pred_grad_mul_offset[:, 0, :, :] = -1.0 * y1  # -y1
        bbox_pred_grad_mul_offset[:, 1, :, :] = -1.0 * x1  # -x1
        bbox_pred_grad_mul_offset[:, 2, :, :] = -1.0 * y1  # -y1
        bbox_pred_grad_mul_offset[:, 4, :, :] = -1.0 * y1  # -y1
        bbox_pred_grad_mul_offset[:, 5, :, :] = x2  # x2
        bbox_pred_grad_mul_offset[:, 7, :, :] = -1.0 * x1  # -x1
        bbox_pred_grad_mul_offset[:, 11, :, :] = x2  # x2
        bbox_pred_grad_mul_offset[:, 12, :, :] = y2  # y2
        bbox_pred_grad_mul_offset[:, 13, :, :] = -1.0 * x1  # -x1
        bbox_pred_grad_mul_offset[:, 14, :, :] = y2  # y2
        bbox_pred_grad_mul_offset[:, 16, :, :] = y2  # y2
        bbox_pred_grad_mul_offset[:, 17, :, :] = x2  # x2
        dcn_offset = bbox_pred_grad_mul_offset - dcn_base_offset

        return dcn_offset    

    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor, bbox_pred_refine: Tensor,
                            labels: Tensor, label_weights: Tensor,
                            bbox_targets: Tensor, assign_metrics: Tensor,
                            stride: List[int]):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Decoded bboxes for each scale
                level with shape (N, num_anchors * 4, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors).
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            assign_metrics (Tensor): Assign metrics with shape
                (N, num_total_anchors).
            stride (List[int]): Downsample stride of the feature map.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.reshape(-1, 4)
        bbox_pred_refine = bbox_pred_refine.reshape(-1, 4)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        assign_metrics = assign_metrics.reshape(-1)
        label_weights = label_weights.reshape(-1)
        targets = (labels, assign_metrics)

        loss_cls = self.loss_cls(
            cls_score, targets, label_weights, avg_factor=1.0)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_bbox_pred_refine = bbox_pred_refine[pos_inds]

            pos_decode_bbox_pred_refine = pos_bbox_pred_refine
            pos_decode_bbox_pred = pos_bbox_pred
            pos_decode_bbox_targets = pos_bbox_targets

            # regression loss
            pos_bbox_weight = assign_metrics[pos_inds]

            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=pos_bbox_weight,
                avg_factor=1.0)
            loss_bbox_refine = self.loss_bbox(
                pos_decode_bbox_pred_refine,
                pos_decode_bbox_targets,
                weight=pos_bbox_weight,
                avg_factor=1.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_bbox_refine = bbox_pred_refine.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, loss_bbox_refine, assign_metrics.sum(), pos_bbox_weight.sum()

    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     bbox_preds_refine: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Decoded box for each scale
                level with shape (N, num_anchors * 4, H, W) in
                [tl_x, tl_y, br_x, br_y] format.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ], 1)
        decoded_bboxes = []
        decoded_bboxes_refine = []
        for anchor, bbox_pred, bbox_pred_refine in zip(anchor_list[0], bbox_preds, bbox_preds_refine):
            anchor = anchor.reshape(-1, 4)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            bbox_pred = distance2bbox(anchor, bbox_pred)
            decoded_bboxes.append(bbox_pred)

            bbox_pred_refine = bbox_pred_refine.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            bbox_pred_refine = distance2bbox(anchor, bbox_pred_refine)
            decoded_bboxes_refine.append(bbox_pred_refine)

        flatten_bboxes = torch.cat(decoded_bboxes, 1)

        cls_reg_targets = self.get_targets(
            flatten_cls_scores,
            flatten_bboxes,
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         assign_metrics_list) = cls_reg_targets

        losses_cls, losses_bbox, losses_bbox_refine, \
            cls_avg_factors, bbox_avg_factors = multi_apply(
                self.loss_by_feat_single,
                cls_scores,
                decoded_bboxes,
                decoded_bboxes_refine,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                assign_metrics_list,
                self.prior_generator.strides)

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        bbox_avg_factor = reduce_mean(
            sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        losses_bbox_refine = list(map(lambda x: x / bbox_avg_factor, losses_bbox_refine))
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_bbox_refine= losses_bbox_refine)

    def get_targets(self,
                    cls_scores: Tensor,
                    bbox_preds: Tensor,
                    anchor_list: List[List[Tensor]],
                    valid_flag_list: List[List[Tensor]],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore: OptInstanceList = None,
                    unmap_outputs=True):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            cls_scores (Tensor): Classification predictions of images,
                a 3D-Tensor with shape [num_imgs, num_priors, num_classes].
            bbox_preds (Tensor): Decoded bboxes predictions of one image,
                a 3D-Tensor with shape [num_imgs, num_priors, 4] in [tl_x,
                tl_y, br_x, br_y] format.
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors. Defaults to True.

        Returns:
            tuple: a tuple containing learning targets.

            - anchors_list (list[list[Tensor]]): Anchors of each level.
            - labels_list (list[Tensor]): Labels of each level.
            - label_weights_list (list[Tensor]): Label weights of each
              level.
            - bbox_targets_list (list[Tensor]): BBox targets of each level.
            - assign_metrics_list (list[Tensor]): alignment metrics of each
              level.
        """
        num_imgs = len(batch_img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs
        # anchor_list: list(b * [-1, 4])
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_assign_metrics) = multi_apply(
             self._get_targets_single,
             cls_scores.detach(),
             bbox_preds.detach(),
             anchor_list,
             valid_flag_list,
             batch_gt_instances,
             batch_img_metas,
             batch_gt_instances_ignore,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None

        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        assign_metrics_list = images_to_levels(all_assign_metrics,
                                               num_level_anchors)

        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, assign_metrics_list)

    def _get_targets_single(self,
                            cls_scores: Tensor,
                            bbox_preds: Tensor,
                            flat_anchors: Tensor,
                            valid_flags: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            cls_scores (list(Tensor)): Box scores for each image.
            bbox_preds (list(Tensor)): Box energies / deltas for each image.
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors. Defaults to True.

        Returns:
            tuple: N is the number of total anchors in the image.

            - anchors (Tensor): All anchors in the image with shape (N, 4).
            - labels (Tensor): Labels of all anchors in the image with shape
              (N,).
            - label_weights (Tensor): Label weights of all anchor in the
              image with shape (N,).
            - bbox_targets (Tensor): BBox targets of all anchors in the
              image with shape (N, 4).
            - norm_alignment_metrics (Tensor): Normalized alignment metrics
              of all priors in the image with shape (N,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        pred_instances = InstanceData(
            scores=cls_scores[inside_flags, :],
            bboxes=bbox_preds[inside_flags, :],
            priors=anchors)

        assign_result = self.assigner.assign(pred_instances, gt_instances,
                                             gt_instances_ignore)

        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        assign_metrics = anchors.new_zeros(
            num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            # point-based
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets

            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        class_assigned_gt_inds = torch.unique(
            sampling_result.pos_assigned_gt_inds)
        for gt_inds in class_assigned_gt_inds:
            gt_class_inds = pos_inds[sampling_result.pos_assigned_gt_inds ==
                                     gt_inds]
            assign_metrics[gt_class_inds] = assign_result.max_overlaps[
                gt_class_inds]

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            assign_metrics = unmap(assign_metrics, num_total_anchors,
                                   inside_flags)
        return (anchors, labels, label_weights, bbox_targets, assign_metrics)

    def get_anchors(self,
                    featmap_sizes: List[tuple],
                    batch_img_metas: List[dict],
                    device: Union[torch.device, str] = 'cuda') \
            -> Tuple[List[List[Tensor]], List[List[Tensor]]]:
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            batch_img_metas (list[dict]): Image meta info.
            device (torch.device or str): Device for returned tensors.
                Defaults to cuda.

        Returns:
            tuple:

            - anchor_list (list[list[Tensor]]): Anchors of each image.
            - valid_flag_list (list[list[Tensor]]): Valid flags of each
              image.
        """
        num_imgs = len(batch_img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device, with_stride=True)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(batch_img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)
        return anchor_list, valid_flag_list