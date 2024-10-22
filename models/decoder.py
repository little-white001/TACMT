import torch
import torch.nn.init as init
from torch import nn
from collections import OrderedDict


from .transformer import TransformerDecoderLayer, TransformerDecoder, get_activation, bias_init_with_prob


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class DeformableDecoder(nn.Module):
    def __init__(self,
                 hidden_dim=256,
                 nhead=8,
                 position_embed_type='sine',
                 feat_channels=[256, 256, 256],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_queries=1,
                 eps=1e-2,
                 num_decoder_layers=6,
                 eval_spatial_size=None,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_decoder_points=4,
                 eval_idx=-1,
                 learnt_init_query=False,
                 # word_attn_args=None,
                 ):
        super().__init__()

        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_spatial_size = eval_spatial_size

        # Transformer module
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)


        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim,)
        )
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])



        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters()

    def _reset_parameters(self):
        # bias = bias_init_with_prob(0.01)


        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for reg_ in  self.dec_bbox_head:
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)

        # linear_init_(self.enc_output[0])
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)



    def _get_encoder_input(self, feats):
        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.cat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = [[int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)]
                for s in self.feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            # grid_y, grid_x = torch.meshgrid(torch.arange(end=h, dtype=dtype), torch.arange(end=w, dtype=dtype), indexing='ij')
            grid_x, grid_y = torch.meshgrid(torch.arange(h, dtype=dtype), torch.arange(w, dtype=dtype)) # old pytorch version
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_WH = torch.tensor([w, h]).to(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            anchors.append(torch.cat([grid_xy, wh], -1).reshape(-1, h * w, 4))

        anchors = torch.cat(anchors, 1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))

        inf_double = torch.tensor(float('inf'), dtype=torch.float).to(device)
        anchors = torch.where(valid_mask, anchors, inf_double)

        return anchors, valid_mask

    def _get_decoder_input(self,
                           memory,
                           spatial_shapes,
                           word_feat):
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

        memory = valid_mask.to(memory.dtype) * memory

        output_memory = self.enc_output(memory)

        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        logits = torch.einsum("bic,btc->bit", memory, word_feat.permute(1,0,2)) # [bs,num_img_tokens,num_text_tokens]
        logits_per_img_feat = logits.max(-1)[0] # [bs,num_img_tokens]
        topk_ind = torch.topk(logits_per_img_feat,self.num_queries,dim=1)[1]

        reference_points_unact = enc_outputs_coord_unact.gather(dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1,enc_outputs_coord_unact.shape[-1])) # [4,1,4]


        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = output_memory.gather(dim=1,
                index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target


        return target, reference_points_unact

    def forward(self, feats, word_feat, word_mask):

        # input projection and embedding
        # memory.shape=[bs,5376,256] spatial_shapes=[[64,64],[32,32],[16,16]] level_start_index[0,4096,5120]
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)
        # [bs,1,hidden_dim] [bs,1,npts]
        memory_chunk = memory.chunk(2, dim=-1)[0] # ori_img_feat
        target, init_ref_points_unact = self._get_decoder_input(memory_chunk, spatial_shapes, word_feat)

        # decoder
        out_bboxes = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.query_pos_head,
            # attn_mask=None,
            # memory_mask=None,
            word_feat=word_feat)

        return out_bboxes

class vg_decoder_wrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        args = cfg.copy()
        decoder_type= args.pop('type')
        self.decoder = _MODULES[decoder_type](**args)


    def forward(self, img_feats, word_feat, word_mask):
        hs = self.decoder(img_feats, word_feat, word_mask)
        return hs


_MODULES = {
    'DeformableDecoder': DeformableDecoder,
}

def build_vg_decoder(args):
    return vg_decoder_wrapper(args.model_config['decoder'])
