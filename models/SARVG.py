import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, get_world_size, is_dist_avail_and_initialized)

from .backbone import build_backbone

from .transformer import build_visual_encoder
from .decoder import build_vg_decoder
from .encoder import build_cross_modal_encoder
from pytorch_pretrained_bert.modeling import BertModel


class SARVG(nn.Module):
    def __init__(self, pretrained_weights, args=None):
        """ Initializes the model."""
        super().__init__()

        # Image feature encoder (CNN + Transformer encoder)
        self.backbone = build_backbone(args)
        self.trans_encoder = build_visual_encoder(args)

        # Text feature encoder (BERT)
        self.bert = BertModel.from_pretrained(args.bert_model) # download bert for the first run
        # self.bert = BertModel.from_pretrained('/root/autodl-tmp/codes/sarvg_latest/pretrained/bert-base-uncased') # using local weights
        self.bert_proj = nn.Linear(args.bert_output_dim, args.hidden_dim)
        self.bert_output_layers = args.bert_output_layers
        for v in self.bert.pooler.parameters():
            v.requires_grad_(False)

        # cross modal encoder
        # self.cross_modal_encoder = build_cross_modal_encoder(args)
        self.cross_modal_encoder = nn.ModuleList()
        for _ in range(3):
            encoder = build_cross_modal_encoder(args)
            self.cross_modal_encoder.append(encoder)

        # multi stage decoder
        self.multi_stage_decoder = build_vg_decoder(args)

        if pretrained_weights:
            self.load_pretrained_weights(pretrained_weights)

    def load_pretrained_weights(self, weights_path):
        def load_weights(module, prefix, weights):
            module_state_dict = module.state_dict()
            update_weights = {}

            for k, v in module_state_dict.items():
                prefixed_key = f"{prefix}.{k}"
                if prefixed_key in weights and weights[prefixed_key].shape == v.shape:
                    update_weights[k] = weights[prefixed_key]
                # else:
                #     print(f"Skipping {k} due to size mismatch or not found in the checkpoint.")
            module.load_state_dict(update_weights, strict=False)

        weights = torch.load(weights_path, map_location='cpu')['ema']['module']
        load_weights(self.backbone[0].body, prefix='backbone', weights=weights)
        load_weights(self.trans_encoder.encoder, prefix='encoder', weights=weights)

    def forward(self, image, image_mask, word_id, word_mask):
        # Image features from cnn
        features, pos = self.backbone(NestedTensor(image, image_mask))
        # decompose
        src=[]
        mask=[]
        for i in range(len(features)):
            srctmp, masktmp=features[i].decompose()
            src.append(srctmp)
            mask.append(masktmp)
        assert mask is not None

        # Image features from transformer
        img_feats = self.trans_encoder(src, mask=mask, pos_embed=pos) ## [[bs,c,64,64],[bs,c,32,32],[bs,c,16,16]]

        # Text features
        word_feat, _ = self.bert(word_id, token_type_ids=None, attention_mask=word_mask)
        word_feat = torch.stack(word_feat[-self.bert_output_layers:], 1).mean(1)
        word_feat = self.bert_proj(word_feat)
        word_feat = word_feat.permute(1, 0, 2) # NxLxC -> LxNxC
        word_mask = ~word_mask

        # [bs, c, h, w] -> [bs, c, hw] -> [hw, bs, c]
        img_feats_flattened = [f.flatten(2).permute(2, 0, 1) for f in img_feats]
        masks_flattened = [m.flatten(1) for m in mask]
        pos_embeds_flattened = [p.flatten(2).permute(2, 0, 1) for p in pos]

        img_feats_cross = [self.cross_modal_encoder[idx](img_feat, mask_flattened, pos_embed, word_feat, word_mask)
                           for idx, (img_feat, mask_flattened, pos_embed) in
                           enumerate(zip(img_feats_flattened, masks_flattened, pos_embeds_flattened))] ## [[4096,bs,c],[1024,bs,c],[256,bs,c]]

        img_feats = [feat.permute(1, 2, 0).view(bs, 2*c, h, w) for feat, (bs, c, h, w) in
                     zip(img_feats_cross, [f.shape for f in img_feats])] # ## [[bs,c,64,64],[bs,c,32,32],[bs,c,16,16]]

        # Multi-stage decode
        outputs_coord = self.multi_stage_decoder(img_feats, word_feat, word_mask)

        out = {'pred_boxes': outputs_coord[-1]}

        if self.training:
            out['aux_outputs'] = [{'pred_boxes': b} for b in outputs_coord[:-1]]
        return out




class VGCriterion(nn.Module):
    """ This class computes the loss for SARVG."""
    def __init__(self, weight_dict, loss_loc, box_xyxy):
        """ Create the criterion.
        Parameters:
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
        """
        super().__init__()
        self.weight_dict = weight_dict

        self.box_xyxy = box_xyxy

        self.loss_map = {'loss_boxes': self.loss_boxes}

        self.loss_loc = self.loss_map[loss_loc]

    def loss_boxes(self, outputs, target_boxes, num_pos):
        """Compute the losses related to the bounding boxes (the L1 regression loss and the GIoU loss)"""
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes'] # [B, #query, 4]
        target_boxes = target_boxes[:, None].expand_as(src_boxes)

        src_boxes = src_boxes.reshape(-1, 4) # [B*#query, 4]
        target_boxes = target_boxes.reshape(-1, 4) #[B*#query, 4]

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['l1'] = loss_bbox.sum() / num_pos

        if not self.box_xyxy:
            src_boxes = box_ops.box_cxcywh_to_xyxy(src_boxes)
            target_boxes = box_ops.box_cxcywh_to_xyxy(target_boxes)
        loss_giou = 1 - box_ops.box_pair_giou(src_boxes, target_boxes)
        losses['giou'] = (loss_giou[:, None]).sum() / num_pos
        return losses


    def forward(self, outputs, targets):
        """ This performs the loss computation.
        """
        gt_boxes = targets['bbox']
        pred_boxes = outputs['pred_boxes']

        losses = {}
        B, Q, _ = pred_boxes.shape
        num_pos = avg_across_gpus(pred_boxes.new_tensor(B*Q))
        loss = self.loss_loc(outputs, gt_boxes, num_pos)
        losses.update(loss)

        # Apply the loss function to the outputs from all the stages
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                l_dict = self.loss_loc(aux_outputs, gt_boxes, num_pos)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format we expect"""
    def __init__(self, box_xyxy=False):
        super().__init__()
        self.bbox_xyxy = box_xyxy

    @torch.no_grad()
    def forward(self, outputs, target_dict):
        """ Perform the computation"""
        rsz_sizes, ratios, orig_sizes = \
            target_dict['size'], target_dict['ratio'], target_dict['orig_size']
        dxdy = None if 'dxdy' not in target_dict else target_dict['dxdy']

        boxes = outputs['pred_boxes']

        assert len(boxes) == len(rsz_sizes)
        assert rsz_sizes.shape[1] == 2

        boxes = boxes.squeeze(1)

        # Convert to absolute coordinates in the original image
        if not self.bbox_xyxy:
            boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        img_h, img_w = rsz_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct
        if dxdy is not None:
            boxes = boxes - torch.cat([dxdy, dxdy], dim=1)
        boxes = boxes.clamp(min=0)
        ratio_h, ratio_w = ratios.unbind(1)
        boxes = boxes / torch.stack([ratio_w, ratio_h, ratio_w, ratio_h], dim=1)
        if orig_sizes is not None:
            orig_h, orig_w = orig_sizes.unbind(1)
            boxes = torch.min(boxes, torch.stack([orig_w, orig_h, orig_w, orig_h], dim=1))

        return boxes


def avg_across_gpus(v, min=1):
    if is_dist_avail_and_initialized():
        torch.distributed.all_reduce(v)
    return torch.clamp(v.float() / get_world_size(), min=min).item()


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x




def build_vgmodel(args):
    device = torch.device(args.device)

    model = SARVG(pretrained_weights=args.load_weights_path, args=args)

    weight_dict = {'loss_cls': 1, 'l1': args.bbox_loss_coef}
    weight_dict['giou'] = args.giou_loss_coef
    weight_dict.update(args.other_loss_coefs)
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = VGCriterion(weight_dict=weight_dict, loss_loc=args.loss_loc, box_xyxy=args.box_xyxy)
    criterion.to(device)

    postprocessor = PostProcess(args.box_xyxy)

    return model, criterion, postprocessor
