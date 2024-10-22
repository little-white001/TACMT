import copy
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


## basic modules
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        if num_layers > 0:
            h = [hidden_dim] * (num_layers - 1)
            self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = []

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


MULTIHEAD_ATTNS = {
    'MultiheadAttention': nn.MultiheadAttention,
}

class CrossModalEncLayer(nn.Module):
    def __init__(self, d_model, img2text_attn1_args=None, img_query_with_pos=True,
                 img2text_attn2_args=None, img2img_attn_args=None, fusion_args=None):
        super().__init__()
        args = img2text_attn1_args.copy()
        self.img2text_attn1 = MULTIHEAD_ATTNS[args.pop('type')](**args)
        self.img_query_with_pos = img_query_with_pos

        self.text_proj = MLP(**fusion_args['text_proj'])
        self.img_proj = MLP(**fusion_args['img_proj'])
        self.scale = Parameter(torch.Tensor([fusion_args.get('scale')]))

        args = img2text_attn2_args.copy()
        self.img2text_attn2 = MULTIHEAD_ATTNS[args.pop('type')](**args)

        args = img2img_attn_args.copy()
        self.img2img_attn = MULTIHEAD_ATTNS[args.pop('type')](**args)

        self.norm_img = nn.LayerNorm(d_model)
        self.norm_modulated_img = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, img_feat, img_key_padding_mask, img_pos,
                word_feat, word_key_padding_mask, word_pos=None):
        orig_img_feat = img_feat

        # visual-linguistic fusion
        img_query = img_feat + img_pos if self.img_query_with_pos else img_feat
        text_info = self.img2text_attn1(
            query=img_query, key=self.with_pos_embed(word_feat, word_pos),
            value=word_feat, key_padding_mask=word_key_padding_mask)[0]

        text_embed = self.text_proj(text_info)
        img_embed = self.img_proj(img_feat)

        distance = F.normalize(img_embed - text_embed, p=1, dim=-1)
        fusion_score = torch.exp(-1 * self.scale * distance)

        # text-guided visual feature
        text_guided_visual_info = self.img2text_attn2(
            query=img_feat, key=self.with_pos_embed(word_feat, word_pos),
            value=word_feat, key_padding_mask=word_key_padding_mask)[0]

        q = k = img_feat + text_guided_visual_info
        modulated_img_feature = self.img2img_attn(query=q, key=k, value=img_feat, key_padding_mask=img_key_padding_mask)[0]

        # selected feature
        fuse_img_feat = (self.norm_img(img_feat) +
                         self.norm_modulated_img(modulated_img_feature)) * fusion_score

        return torch.cat([orig_img_feat, fuse_img_feat], dim=-1)
        # return orig_img_feat + fuse_img_feat

class CrossModalEncoder(nn.Module):
    def __init__(self, num_layers=1, layer=None):
        super().__init__()
        args = layer.copy()
        layer_type = args.pop('type')
        encoder_layer = _MODULES[layer_type](**args)
        self.encoder_layers = _get_clones(encoder_layer, num_layers)

    def forward(self, img_feat, img_key_padding_mask=None, pos=None,
                word_feat=None, word_key_padding_mask=None):

        # intermediate = []
        # Encode discriminative features
        for layer in self.encoder_layers:
            img_feat = layer(img_feat, img_key_padding_mask, pos,
                             word_feat, word_key_padding_mask, None)

        return img_feat

_MODULES ={
    'CrossModalEncoder': CrossModalEncoder,
    'CrossModalEncLayer': CrossModalEncLayer,
}


class vg_encoder_wrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        args = cfg.copy()
        encoder_type = args.pop('type')
        self.encoder = _MODULES[encoder_type](**args)


    def forward(self, img_feat, mask, pos_embed, word_feat, word_mask):
        cross_modal_feat = self.encoder(img_feat, mask, pos_embed, word_feat, word_mask)

        return cross_modal_feat

def build_cross_modal_encoder(args):
    return vg_encoder_wrapper(args.model_config['encoder'])





