import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pdb import set_trace as stop
import os, random
import warnings
from utils.func import convert_image_to_rgb, Uint16ToFloatTransform, CustomToTensor
from transformers import BertTokenizer
from dataset.common import CGCommon_T2I, collate_fn_T2I, CGCommon_T2I_L
from functools import partial
import json
from torch.utils.data.distributed import DistributedSampler
warnings.filterwarnings("ignore")

def get_data(args):
    dataset = args.dataset
    meta_root = args.meta_root
    data_root = args.dataroot
    batch_size = args.batch_size

    meta_file = args.meta_file

    rescale = args.scale_size
    random_crop = args.crop_size
    workers = args.workers

    BEN_normTransform = transforms.Normalize(mean=[340.76769064, 429.9430203, 614.21682446,
                     590.23569706, 950.68368468, 1792.46290469,
                    2075.46795189,2218.94553375,2266.46036911,
                    2246.0605464,1594.42694882,1009.32729131],
                                             std=[554.81258967,572.41639287,582.87945694,
                675.88746967,729.89827633,1096.01480586,
                     1273.45393088,1365.45589904,1356.13789355,
                     1302.3292881,1079.19066363,818.86747235]) # BigEarthNet
    scale_size = rescale
    crop_size = random_crop
    if not args.test_batch_size:
        args.test_batch_size = 1

    BEN_trainTransform = transforms.Compose([
                                        Uint16ToFloatTransform(),
                                        transforms.Resize((scale_size, scale_size)),
                                        transforms.RandomCrop((crop_size, crop_size)),
                                        transforms.RandomHorizontalFlip(),
                                        ])



    BEN_testTransform = transforms.Compose([Uint16ToFloatTransform(),
                                        transforms.Resize((scale_size, scale_size)),
                                        transforms.CenterCrop(crop_size),
                                        ])

    if dataset == 'DLRSD':
        args.image_root = "/mnt/data/jiangfanzhuo/DLRSD_CGSIR/images/"
        args.train_datas_path = "/mnt/data/jiangfanzhuo/DLRSD_CGSIR/train_json"
        args.test_datas_path = "/mnt/data/jiangfanzhuo/DLRSD_CGSIR/test_json"
        with open(args.train_datas_path, "r", encoding="utf-8") as file:
            train_datas = json.load(file)
        with open(args.test_datas_path, "r", encoding="utf-8") as file:
            test_datas = json.load(file)
        normTransform = transforms.Normalize(mean=train_datas["norm"]["mean"], std=train_datas["norm"]["std"])  # RGB
        pre_trainTransform = transforms.Compose([transforms.ToTensor(),
                                                 normTransform])
        trainTransform = transforms.Compose([transforms.RandomResizedCrop((crop_size, crop_size)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.RandomRotation(degrees=90)
                                             ])
        testTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                            transforms.CenterCrop(crop_size),
                                            transforms.ToTensor(),
                                            normTransform
                                            ])

        query_dataset = CGCommon_T2I(args.image_root, test_datas["data"], split="query", transform=testTransform, pre_transform=None)
        db_dataset = CGCommon_T2I(args.image_root, train_datas["data"], split="db", transform=trainTransform, pre_transform=pre_trainTransform)
        if args.test_noise:
            gallery_dataset = CGCommon_T2I(args.image_root, test_datas["data"], split="noise_db", transform=testTransform, pre_transform=None)
        else:
            gallery_dataset = CGCommon_T2I(args.image_root, test_datas["data"], split="db", transform=testTransform, pre_transform=None)
        tokenizer = BertTokenizer.from_pretrained(args.text_encoder_model_path)

        sampler_db = DistributedSampler(db_dataset, shuffle=True, drop_last=True)

        query_loader = DataLoader(query_dataset, batch_size=args.test_batch_size, num_workers=workers, shuffle=False,
                                  pin_memory=True, collate_fn=partial(collate_fn_T2I, tokenizer=tokenizer))
        db_loader = DataLoader(db_dataset, batch_size=batch_size, num_workers=workers,
                               pin_memory=True, collate_fn=partial(collate_fn_T2I, tokenizer=tokenizer),
                               sampler=sampler_db)
        gallery_loader = DataLoader(gallery_dataset, batch_size=args.test_batch_size,
                                    num_workers=workers, pin_memory=True, shuffle=False,
                                    collate_fn=partial(collate_fn_T2I, tokenizer=tokenizer))

    elif dataset == "MultiScene-Clean":
        args.image_root = "/mnt/data/jiangfanzhuo/MSC_CGSIR/images/"
        args.train_datas_path = "/mnt/data/jiangfanzhuo/MSC_CGSIR/train_json"
        args.test_datas_path = "/mnt/data/jiangfanzhuo/MSC_CGSIR/test_json"
        with open(args.train_datas_path, "r", encoding="utf-8") as file:
            train_datas = json.load(file)
        with open(args.test_datas_path, "r", encoding="utf-8") as file:
            test_datas = json.load(file)
        normTransform = transforms.Normalize(mean=train_datas["norm"]["mean"], std=train_datas["norm"]["std"])  # RGB
        pre_trainTransform = transforms.Compose([transforms.ToTensor(),
                                                 normTransform])
        trainTransform = transforms.Compose([transforms.RandomResizedCrop((crop_size, crop_size)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.RandomRotation(degrees=90)
                                             ])
        testTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                            transforms.CenterCrop(crop_size),
                                            transforms.ToTensor(),
                                            normTransform
                                            ])

        query_dataset = CGCommon_T2I(args.image_root, test_datas["data"], split="query", transform=testTransform,
                                     pre_transform=None)
        db_dataset = CGCommon_T2I(args.image_root, train_datas["data"], split="db", transform=trainTransform,
                                  pre_transform=pre_trainTransform)
        if args.test_noise:
            gallery_dataset = CGCommon_T2I(args.image_root, test_datas["data"], split="noise_db",
                                           transform=testTransform, pre_transform=None, noise_rate=args.noise_rate)
        else:
            gallery_dataset = CGCommon_T2I(args.image_root, test_datas["data"], split="db", transform=testTransform,
                                           pre_transform=None)
        tokenizer = BertTokenizer.from_pretrained(args.text_encoder_model_path)

        query_loader = DataLoader(query_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=workers,
                                  pin_memory=True, collate_fn=partial(collate_fn_T2I, tokenizer=tokenizer))
        gallery_loader = DataLoader(gallery_dataset, batch_size=args.test_batch_size, shuffle=False,
                                    num_workers=workers, pin_memory=True,
                                    collate_fn=partial(collate_fn_T2I, tokenizer=tokenizer))
        is_inference = False
        if "is_inference" in args:
            is_inference = args.is_inference
        if not is_inference:
            sampler_db = DistributedSampler(db_dataset, shuffle=True, drop_last=True)
            db_loader = DataLoader(db_dataset, batch_size=batch_size, num_workers=workers,
                                   pin_memory=True, collate_fn=partial(collate_fn_T2I, tokenizer=tokenizer),
                                   sampler=sampler_db)
            return query_loader, gallery_loader, db_loader, sampler_db
        else:
            return query_loader, gallery_loader

    elif dataset == "DOTA":
        args.image_root = "/mnt/data/jiangfanzhuo/DOTA_FGSIR/images/"
        args.train_datas_path = "/mnt/data/jiangfanzhuo/DOTA_FGSIR/train_json"
        args.test_datas_path = "/mnt/data/jiangfanzhuo/DOTA_FGSIR/test_json"
        with open(args.train_datas_path, "r", encoding="utf-8") as file:
            train_datas = json.load(file)
        with open(args.test_datas_path, "r", encoding="utf-8") as file:
            test_datas = json.load(file)
        normTransform = transforms.Normalize(mean=train_datas["norm"]["mean"], std=train_datas["norm"]["std"])  # RGB
        trainTransform = transforms.Compose([transforms.Resize((crop_size, crop_size)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.RandomRotation(degrees=90),
                                             transforms.ToTensor(),
                                             normTransform
                                             ])
        testTransform = transforms.Compose([transforms.Resize((crop_size, crop_size)),
                                            transforms.ToTensor(),
                                            normTransform
                                            ])

        query_dataset = CGCommon_T2I_L(args.image_root, test_datas["data"], split="query", transform=testTransform)
        db_dataset = CGCommon_T2I_L(args.image_root, train_datas["data"], split="db", transform=trainTransform)
        if args.test_noise:
            gallery_dataset = CGCommon_T2I_L(args.image_root, test_datas["data"], split="noise_db",
                                           transform=testTransform, noise_rate=args.noise_rate)
        else:
            gallery_dataset = CGCommon_T2I_L(args.image_root, test_datas["data"], split="db", transform=testTransform)
        tokenizer = BertTokenizer.from_pretrained(args.text_encoder_model_path)

        query_loader = DataLoader(query_dataset, batch_size=args.test_batch_size, num_workers=workers, shuffle=False,
                                  pin_memory=True, collate_fn=partial(collate_fn_T2I, tokenizer=tokenizer))
        gallery_loader = DataLoader(gallery_dataset, batch_size=args.test_batch_size,
                                    num_workers=workers, pin_memory=True, shuffle=False,
                                    collate_fn=partial(collate_fn_T2I, tokenizer=tokenizer))

        is_inference = False
        if "is_inference" in args:
            is_inference = args.is_inference
        if not is_inference:
            sampler_db = DistributedSampler(db_dataset, shuffle=True, drop_last=True)
            db_loader = DataLoader(db_dataset, batch_size=batch_size, num_workers=workers,
                                   pin_memory=True, collate_fn=partial(collate_fn_T2I, tokenizer=tokenizer),
                                   sampler=sampler_db)
            return query_loader, gallery_loader, db_loader, sampler_db
        else:
            return query_loader, gallery_loader

    elif dataset == "RICE-CG":
        args.image_root = "/mnt/data/jiangfanzhuo/RICE_CGSIR/images/"
        args.train_datas_path = "/mnt/data/jiangfanzhuo/RICE_CGSIR/train_json"
        args.test_datas_path = "/mnt/data/jiangfanzhuo/RICE_CGSIR/test_json"
        with open(args.train_datas_path, "r", encoding="utf-8") as file:
            train_datas = json.load(file)
        with open(args.test_datas_path, "r", encoding="utf-8") as file:
            test_datas = json.load(file)
        normTransform = transforms.Normalize(mean=train_datas["norm"]["mean"], std=train_datas["norm"]["std"])  # RGB
        trainTransform = transforms.Compose([transforms.Resize((crop_size, crop_size)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.RandomRotation(degrees=90),
                                             transforms.ToTensor(),
                                             normTransform
                                             ])
        testTransform = transforms.Compose([transforms.Resize((crop_size, crop_size)),
                                            transforms.ToTensor(),
                                            normTransform
                                            ])

        query_dataset = CGCommon_T2I_L(args.image_root, test_datas["data"], split="query", transform=testTransform)
        db_dataset = CGCommon_T2I_L(args.image_root, train_datas["data"], split="db", transform=trainTransform)
        if args.test_noise:
            gallery_dataset = CGCommon_T2I_L(args.image_root, test_datas["data"], split="noise_db",
                                           transform=testTransform)
        else:
            gallery_dataset = CGCommon_T2I_L(args.image_root, train_datas["data"], split="db", transform=testTransform)
        tokenizer = BertTokenizer.from_pretrained(args.text_encoder_model_path)

        query_loader = DataLoader(query_dataset, batch_size=args.test_batch_size, num_workers=workers, shuffle=False,
                                  pin_memory=True, collate_fn=partial(collate_fn_T2I, tokenizer=tokenizer))
        gallery_loader = DataLoader(gallery_dataset, batch_size=args.test_batch_size,
                                    num_workers=workers, pin_memory=True, shuffle=False,
                                    collate_fn=partial(collate_fn_T2I, tokenizer=tokenizer))

        is_inference = False
        if "is_inference" in args:
            is_inference = args.is_inference
        if not is_inference:
            sampler_db = DistributedSampler(db_dataset, shuffle=True, drop_last=True)
            db_loader = DataLoader(db_dataset, batch_size=batch_size, num_workers=workers,
                                   pin_memory=True, collate_fn=partial(collate_fn_T2I, tokenizer=tokenizer),
                                   sampler=sampler_db)
            return query_loader, gallery_loader, db_loader, sampler_db
        else:
            return query_loader, gallery_loader
    elif dataset == "RICE-SIR":
        args.image_root = "/mnt/data/jiangfanzhuo/RICE_SIR/images/"
        args.train_datas_path = "/mnt/data/jiangfanzhuo/RICE_SIR/train_json"
        args.test_datas_path = "/mnt/data/jiangfanzhuo/RICE_SIR/test_json"
        with open(args.train_datas_path, "r", encoding="utf-8") as file:
            train_datas = json.load(file)
        with open(args.test_datas_path, "r", encoding="utf-8") as file:
            test_datas = json.load(file)
        normTransform = transforms.Normalize(mean=train_datas["norm"]["mean"], std=train_datas["norm"]["std"])  # RGB
        trainTransform = transforms.Compose([transforms.Resize((crop_size, crop_size)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.RandomRotation(degrees=90),
                                             transforms.ToTensor(),
                                             normTransform
                                             ])
        testTransform = transforms.Compose([transforms.Resize((crop_size, crop_size)),
                                            transforms.ToTensor(),
                                            normTransform
                                            ])

        query_dataset = CGCommon_T2I_L(args.image_root, test_datas["data"], split="query", transform=testTransform)
        db_dataset = CGCommon_T2I_L(args.image_root, train_datas["data"], split="db", transform=trainTransform)
        eval_dataset = CGCommon_T2I_L(args.image_root, test_datas["data"], split="db", transform=trainTransform)
        if args.test_noise:
            gallery_dataset = CGCommon_T2I_L(args.image_root, test_datas["data"], split="noise_db",
                                           transform=testTransform)
        else:
            gallery_dataset = CGCommon_T2I_L(args.image_root, test_datas["data"], split="db", transform=testTransform)
        tokenizer = BertTokenizer.from_pretrained(args.text_encoder_model_path)

        query_loader = DataLoader(query_dataset, batch_size=args.test_batch_size, num_workers=workers, shuffle=False,
                                  pin_memory=True, collate_fn=partial(collate_fn_T2I, tokenizer=tokenizer))
        gallery_loader = DataLoader(gallery_dataset, batch_size=args.test_batch_size,
                                    num_workers=workers, pin_memory=True, shuffle=False,
                                    collate_fn=partial(collate_fn_T2I, tokenizer=tokenizer))

        is_inference = False
        if "is_inference" in args:
            is_inference = args.is_inference
        if not is_inference:
            sampler_db = DistributedSampler(db_dataset, shuffle=True, drop_last=True)
            db_loader = DataLoader(db_dataset, batch_size=batch_size, num_workers=workers,
                                   pin_memory=True, collate_fn=partial(collate_fn_T2I, tokenizer=tokenizer),
                                   sampler=sampler_db)

            sampler_eval = DistributedSampler(eval_dataset, shuffle=False, drop_last=False)
            eval_loader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=workers,
                                   pin_memory=True, collate_fn=partial(collate_fn_T2I, tokenizer=tokenizer),
                                   sampler=sampler_eval)
            return query_loader, gallery_loader, db_loader, sampler_db, eval_loader, sampler_eval
        else:
            return query_loader, gallery_loader
    elif dataset == "Cloud-SIR":
        args.image_root = "/mnt/data/jiangfanzhuo/Cloud_SIR/images/"
        args.train_datas_path = "/mnt/data/jiangfanzhuo/Cloud_SIR/train_json"
        args.test_datas_path = "/mnt/data/jiangfanzhuo/Cloud_SIR/test_json"
        with open(args.train_datas_path, "r", encoding="utf-8") as file:
            train_datas = json.load(file)
        with open(args.test_datas_path, "r", encoding="utf-8") as file:
            test_datas = json.load(file)
        normTransform = transforms.Normalize(mean=train_datas["norm"]["mean"], std=train_datas["norm"]["std"])  # RGB
        trainTransform = transforms.Compose([transforms.Resize((crop_size, crop_size)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.RandomRotation(degrees=90),
                                             transforms.ToTensor(),
                                             normTransform
                                             ])
        testTransform = transforms.Compose([transforms.Resize((crop_size, crop_size)),
                                            transforms.ToTensor(),
                                            normTransform
                                            ])

        query_dataset = CGCommon_T2I_L(args.image_root, test_datas["data"], split="query", transform=testTransform)
        db_dataset = CGCommon_T2I_L(args.image_root, train_datas["data"], split="db", transform=trainTransform)
        eval_dataset = CGCommon_T2I_L(args.image_root, test_datas["data"], split="db", transform=trainTransform)
        if args.test_noise:
            gallery_dataset = CGCommon_T2I_L(args.image_root, test_datas["data"], split="noise_db", noise_rate=args.noise_rate,
                                           transform=testTransform)
        else:
            gallery_dataset = CGCommon_T2I_L(args.image_root, test_datas["data"], split="db", transform=testTransform)
        tokenizer = BertTokenizer.from_pretrained(args.text_encoder_model_path)

        query_loader = DataLoader(query_dataset, batch_size=args.test_batch_size, num_workers=workers, shuffle=False,
                                  pin_memory=True, collate_fn=partial(collate_fn_T2I, tokenizer=tokenizer))
        gallery_loader = DataLoader(gallery_dataset, batch_size=args.test_batch_size,
                                    num_workers=workers, pin_memory=True, shuffle=False,
                                    collate_fn=partial(collate_fn_T2I, tokenizer=tokenizer))

        is_inference = False
        if "is_inference" in args:
            is_inference = args.is_inference
        if not is_inference:
            sampler_db = DistributedSampler(db_dataset, shuffle=True, drop_last=True)
            db_loader = DataLoader(db_dataset, batch_size=batch_size, num_workers=workers,
                                   pin_memory=True, collate_fn=partial(collate_fn_T2I, tokenizer=tokenizer),
                                   sampler=sampler_db)
            sampler_eval = DistributedSampler(eval_dataset, shuffle=False, drop_last=False)
            eval_loader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=workers,
                                     pin_memory=True, collate_fn=partial(collate_fn_T2I, tokenizer=tokenizer),
                                     sampler=sampler_eval)
            return query_loader, gallery_loader, db_loader, sampler_db
        else:
            return query_loader, gallery_loader
    elif dataset == "CUHK-SIR":
        args.image_root = "/mnt/data/jiangfanzhuo/CUHK_SIR/images/"
        args.train_datas_path = "/mnt/data/jiangfanzhuo/CUHK_SIR/train_json"
        args.test_datas_path = "/mnt/data/jiangfanzhuo/CUHK_SIR/test_json"
        with open(args.train_datas_path, "r", encoding="utf-8") as file:
            train_datas = json.load(file)
        with open(args.test_datas_path, "r", encoding="utf-8") as file:
            test_datas = json.load(file)
        normTransform = transforms.Normalize(mean=train_datas["norm"]["mean"], std=train_datas["norm"]["std"])  # RGB
        trainTransform = transforms.Compose([transforms.Resize((crop_size, crop_size)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.RandomRotation(degrees=90),
                                             transforms.ToTensor(),
                                             normTransform
                                             ])
        testTransform = transforms.Compose([transforms.Resize((crop_size, crop_size)),
                                            transforms.ToTensor(),
                                            normTransform
                                            ])

        query_dataset = CGCommon_T2I_L(args.image_root, test_datas["data"], split="query", transform=testTransform)
        db_dataset = CGCommon_T2I_L(args.image_root, train_datas["data"], split="db", transform=trainTransform)
        eval_dataset = CGCommon_T2I_L(args.image_root, test_datas["data"], split="db", transform=trainTransform)
        if args.test_noise:
            gallery_dataset = CGCommon_T2I_L(args.image_root, test_datas["data"], split="noise_db",
                                           transform=testTransform)
        else:
            gallery_dataset = CGCommon_T2I_L(args.image_root, test_datas["data"], split="db", transform=testTransform)
        tokenizer = BertTokenizer.from_pretrained(args.text_encoder_model_path)

        query_loader = DataLoader(query_dataset, batch_size=args.test_batch_size, num_workers=workers, shuffle=False,
                                  pin_memory=True, collate_fn=partial(collate_fn_T2I, tokenizer=tokenizer))
        gallery_loader = DataLoader(gallery_dataset, batch_size=args.test_batch_size,
                                    num_workers=workers, pin_memory=True, shuffle=False,
                                    collate_fn=partial(collate_fn_T2I, tokenizer=tokenizer))

        is_inference = False
        if "is_inference" in args:
            is_inference = args.is_inference
        if not is_inference:
            sampler_db = DistributedSampler(db_dataset, shuffle=True, drop_last=True)
            db_loader = DataLoader(db_dataset, batch_size=batch_size, num_workers=workers,
                                   pin_memory=True, collate_fn=partial(collate_fn_T2I, tokenizer=tokenizer),
                                   sampler=sampler_db)
            sampler_eval = DistributedSampler(eval_dataset, shuffle=False, drop_last=False)
            eval_loader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=workers,
                                     pin_memory=True, collate_fn=partial(collate_fn_T2I, tokenizer=tokenizer),
                                     sampler=sampler_eval)
            return query_loader, gallery_loader, db_loader, sampler_db, eval_loader, sampler_eval
        else:
            return query_loader, gallery_loader
    else:
        print('no dataset avail')
        exit(0)

    return query_loader, gallery_loader, db_loader, sampler_db
