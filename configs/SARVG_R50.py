dataset='sarvg'
data_root='path to your file'
split_root='path to your file'
output_dir='./work_dirs/exp1'
img_size=512
epochs=90
lr_drop=60
checkpoint_step=10
enc_layers=1
backbone=resnet50
# bbox_loss_coef=5
# giou_loss_coef=5
freeze_modules=['backbone', 'bert']
load_weights_path='./pretrained/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth'


model_config = dict(
    encoder=dict(
        type='CrossModalEncoder',
        num_layers=1,
        layer=dict(
            type='CrossModalEncLayer',
            d_model=256,
            img_query_with_pos=True,
            img2text_attn1_args=dict(
                type='MultiheadAttention',
                embed_dim=256, num_heads=8, dropout=0.1
            ),
            img2text_attn2_args=dict(
                type='MultiheadAttention',
                embed_dim=256, num_heads=8, dropout=0.1
            ),
            img2img_attn_args=dict(
                type='MultiheadAttention',
                embed_dim=256, num_heads=8, dropout=0.1,
            ),
            fusion_args=dict(
                text_proj=dict(input_dim=256, hidden_dim=256, output_dim=256, num_layers=1),
                img_proj=dict(input_dim=256, hidden_dim=256, output_dim=256, num_layers=1),
                scale=1.0,
            ),
        )
    ),
    decoder=dict(
        type='DeformableDecoder',
        hidden_dim=256,
        nhead=8,
        position_embed_type='sine',
        feat_channels=[256, 256, 256],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_queries=1,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        num_decoder_points=4,
        learnt_init_query=False,
    )
)