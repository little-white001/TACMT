from .dataset import VGDataset


def build_dataset(test, args):
    if test:
        return VGDataset(data_root=args.data_root,
                         split_root=args.split_root,
                         dataset=args.dataset,
                         split=args.test_split,
                         test=True,
                         transforms=args.test_transforms,
                         max_query_len=args.max_query_len,
                         bert_mode=args.bert_token_mode)
    else:
        return VGDataset(data_root=args.data_root,
                          split_root=args.split_root,
                          dataset=args.dataset,
                          split='train',
                          transforms=args.train_transforms,
                          max_query_len=args.max_query_len,
                          bert_mode=args.bert_token_mode)



'''transforms for sarvg'''
train_transforms = [
    dict(type='RandomResize', sizes=[320, 352, 384, 416, 448, 480, 512]), # imsize - 32*i for i in range(0,7)
    dict(type='ToTensor', keys=[]),
    dict(type='NormalizeAndPad', size=512, aug_translate=False)
]

test_transforms = [
    dict(type='RandomResize', sizes=[512], record_resize_info=True),
    dict(type='ToTensor', keys=[]),
    dict(type='NormalizeAndPad', size=512, center_place=True)
]


# '''transforms for refcoco unc'''
# train_transforms = [
#     dict(type='RandomResize', sizes=[448, 480, 512, 544, 576, 608, 640]), # imsize - 32*i for i in range(0,7)
#     dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
#     dict(type='ToTensor', keys=[]),
#     dict(type='NormalizeAndPad', size=640, aug_translate=False)
# ]
#
# test_transforms = [
#     dict(type='RandomResize', sizes=[640], record_resize_info=True),
#     dict(type='ToTensor', keys=[]),
#     dict(type='NormalizeAndPad', size=640, center_place=True)
# ]

# ''' transforms for rsvg'''
# train_transforms = [
#     dict(type='RandomResize', sizes=[640]),
#     dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
#     dict(type='ToTensor', keys=[]),
#     dict(type='NormalizeAndPad', size=640, aug_translate=False)
# ]
#
# test_transforms = [
#     dict(type='RandomResize', sizes=[640], record_resize_info=True),
#     dict(type='ToTensor', keys=[]),
#     dict(type='NormalizeAndPad', size=640, center_place=True)
# ]

