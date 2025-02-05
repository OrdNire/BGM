import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pdb import set_trace as stop
import os, random
import warnings
from utils.func import convert_image_to_rgb, Uint16ToFloatTransform, CustomToTensor
from dataset.dataset_DLRSD import DLRSD, DLRSD_W_noise
from dataset.dataset_MSC import MSC
from dataset.dataset_AID import MLAID
from dataset.dataset_BEN import BEN
from dataset.dataset_WHDLD import WHDLD
from dataset.dataset_FGSC_23 import FGSC_23
from dataset.dataset_PUAS import PlanetUAS
from dataset.dataset_MLRSNet import MLRSNet
from dataset.dataset_SCSShip import SCSShip
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

    normTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # RGB
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

    pre_trainTransform = transforms.Compose([transforms.ToTensor(),
                                             normTransform])
    trainTransform = transforms.Compose([transforms.RandomResizedCrop((scale_size, scale_size)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomRotation(degrees=30)
                                         ])

    BEN_trainTransform = transforms.Compose([
                                        Uint16ToFloatTransform(),
                                        transforms.Resize((scale_size, scale_size)),
                                        transforms.RandomCrop((crop_size, crop_size)),
                                        transforms.RandomHorizontalFlip(),
                                        ])

    pre_testTransform = None
    testTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                        transforms.CenterCrop(crop_size),
                                        transforms.ToTensor(),
                                        normTransform
                                        ])

    BEN_testTransform = transforms.Compose([Uint16ToFloatTransform(),
                                        transforms.Resize((scale_size, scale_size)),
                                        transforms.CenterCrop(crop_size),
                                        ])

    if dataset == 'DLRSD':
        meta_dir = os.path.join(meta_root, "DLRSD")
        data_dir = os.path.join(data_root, "DLRSD/DLRSD", "UCMerced_LandUse/UCMerced_LandUse/Images")
        noise_data_dir = os.path.join(data_root, "DLRSD", f"noise_images_{str(args.noise_rate)}")
        query_dataset = DLRSD_W_noise(meta_dir, data_dir, noise_data_dir, "query", testTransform, meta_file=meta_file, pre_transform=pre_testTransform)
        gallery_dataset = DLRSD_W_noise(meta_dir, data_dir, noise_data_dir, "gallery", testTransform, meta_file=meta_file, pre_transform=pre_testTransform)
        # train_gallery_dataset = DLRSD_W_noise(meta_dir, data_dir, noise_data_dir, "gallery", trainTransform, meta_file=meta_file, pre_transform=pre_trainTransform)
        db_dataset = DLRSD_W_noise(meta_dir, data_dir, noise_data_dir, "db", trainTransform, meta_file=meta_file, pre_transform=pre_trainTransform)
    elif dataset == "MultiScene-Clean":
        trainTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                             transforms.RandomCrop((crop_size, crop_size)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             normTransform
                                             ])
        testTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                            transforms.CenterCrop(crop_size),
                                            transforms.ToTensor(),
                                            normTransform
                                            ])

        meta_dir = os.path.join(meta_root, "MultiScene-Clean")
        data_dir = os.path.join(data_root, "MultiScene-Clean", "MultiScene-Clean/images/images")
        query_dataset = MSC(meta_dir, data_dir, None, "query", testTransform, meta_file=meta_file)
        gallery_dataset = MSC(meta_dir, data_dir, None, "gallery", testTransform, meta_file=meta_file)
        train_gallery_dataset = MSC(meta_dir, data_dir, None, "gallery", trainTransform,
                                      meta_file=meta_file)
        db_dataset = MSC(meta_dir, data_dir, None, "db", trainTransform, meta_file=meta_file)
    elif dataset == "ML-AID":
        meta_dir = os.path.join(meta_root, "ML-AID")
        data_dir = os.path.join(data_root, "ML-AID/ML-AID/image")
        noise_data_dir = os.path.join(data_root, "ML-AID", f"noise_images_{str(args.noise_rate)}")
        query_dataset = MLAID(meta_dir, os.path.join(data_dir, "images_test"), noise_data_dir, "query", testTransform, meta_file=meta_file, pre_transform=pre_testTransform)
        gallery_dataset = MLAID(meta_dir, os.path.join(data_dir, "images_tr"), noise_data_dir, "gallery", testTransform, meta_file=meta_file, pre_transform=pre_testTransform)
        train_gallery_dataset = MLAID(meta_dir, os.path.join(data_dir, "images_tr"), noise_data_dir, "gallery", trainTransform,
                                      meta_file=meta_file, pre_transform=pre_trainTransform)
        db_dataset = MLAID(meta_dir, os.path.join(data_dir, "images_tr"), noise_data_dir, "db", trainTransform, meta_file=meta_file, pre_transform=pre_trainTransform)
    elif dataset == "BigEarthNet-19":
        meta_dir = os.path.join(meta_root, "BigEarthNet-19")
        data_dir = os.path.join(data_root, "BigEarthNet-19/BigEarthNet-19/BigEarthNet-S2")
        db_npmem_file = os.path.join(meta_dir, "BEN_subset_db.dat")
        query_npmem_file = os.path.join(meta_dir, "BEN_subset_query.dat")
        noise_files = {"noise_npmem_file": os.path.join(meta_dir, "BEN_subset_noise_db.dat"), "noise_level_file": "BEN_subset_db.pkl"}
        query_dataset = BEN(meta_dir, data_dir,"query", BEN_testTransform,
                              meta_file=meta_file)
        gallery_dataset = BEN(meta_dir, data_dir,"gallery", BEN_testTransform,
                                meta_file=meta_file)
        train_gallery_dataset = BEN(meta_dir, data_dir,"gallery", BEN_trainTransform,
                                      meta_file=meta_file)
        db_dataset = BEN(meta_dir,  data_dir,"db", BEN_trainTransform, meta_file=meta_file)
    elif dataset == "WHDLD":
        meta_dir = os.path.join(meta_root, "WHDLD")
        data_dir = os.path.join(data_root, "WHDLD/WHDLD", "Images")
        noise_data_dir = os.path.join(data_root, "WHDLD", f"noise_images_{str(args.noise_rate)}")
        query_dataset = WHDLD(meta_dir, data_dir, noise_data_dir, "query", testTransform, meta_file=meta_file, pre_transform=pre_testTransform)
        gallery_dataset = WHDLD(meta_dir, data_dir, noise_data_dir, "gallery", testTransform, meta_file=meta_file, pre_transform=pre_testTransform)
        train_gallery_dataset = WHDLD(meta_dir, data_dir, noise_data_dir, "gallery", trainTransform,
                                      meta_file=meta_file, pre_transform=pre_trainTransform)
        db_dataset = WHDLD(meta_dir, data_dir, noise_data_dir, "db", trainTransform, meta_file=meta_file, pre_transform=pre_trainTransform)
    elif dataset == "FGSC-23":

        trainTransform = transforms.Compose([transforms.RandomResizedCrop((crop_size, crop_size)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.RandomRotation(degrees=30),
                                             transforms.ToTensor(),
                                             normTransform
                                             ])

        testTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                        transforms.CenterCrop(crop_size),
                                        transforms.ToTensor(),
                                        normTransform
                                        ])

        meta_dir = os.path.join(meta_root, "FGSC-23")
        data_dir = os.path.join(data_root, "FGSC-23/FGSC-23")
        noise_data_dir = "/mnt/data/###/noise_data/FGSC-23/noise_images_5.0"
        query_dataset = FGSC_23(meta_dir, os.path.join(data_dir, "test"), noise_data_dir, "query", testTransform, meta_file=meta_file)
        gallery_dataset = FGSC_23(meta_dir, os.path.join(data_dir, "train"), noise_data_dir, "gallery", testTransform, meta_file=meta_file)
        train_gallery_dataset = FGSC_23(meta_dir, os.path.join(data_dir, "train"), noise_data_dir, "gallery", trainTransform,
                                      meta_file=meta_file)
        db_dataset = FGSC_23(meta_dir, os.path.join(data_dir, "train"), noise_data_dir, "db", trainTransform, meta_file=meta_file)
    elif dataset == "PlanetUAS":
        meta_dir = os.path.join(meta_root, "PlanetUAS")
        data_dir = os.path.join(data_root, "PlanetUAS/PlanetUAS", "Images")
        query_dataset = PlanetUAS(meta_dir, data_dir, "query", testTransform, meta_file=meta_file)
        gallery_dataset = PlanetUAS(meta_dir, data_dir, "gallery", testTransform, meta_file=meta_file)
        train_gallery_dataset = PlanetUAS(meta_dir, data_dir, "gallery", trainTransform,
                                      meta_file=meta_file)
        db_dataset = PlanetUAS(meta_dir, data_dir, "db", trainTransform, meta_file=meta_file)
    elif dataset == "MLRSNet":
        trainTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                             transforms.RandomCrop((crop_size, crop_size)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             normTransform
                                             ])
        testTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                            transforms.CenterCrop(crop_size),
                                            transforms.ToTensor(),
                                            normTransform
                                            ])

        meta_dir = os.path.join(meta_root, "MLRSNet")
        data_dir = os.path.join(data_root, "MLRSNet", "MLRSNet/Images")
        query_dataset = MLRSNet(meta_dir, data_dir, None, "query", testTransform, meta_file=meta_file)
        gallery_dataset = MLRSNet(meta_dir, data_dir, None, "gallery", testTransform, meta_file=meta_file)
        train_gallery_dataset = MLRSNet(meta_dir, data_dir, None, "gallery", trainTransform,
                                    meta_file=meta_file)
        db_dataset = MLRSNet(meta_dir, data_dir, None, "db", trainTransform, meta_file=meta_file)
    elif dataset == "SCSShip":
        trainTransform = transforms.Compose([transforms.RandomResizedCrop((crop_size, crop_size)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.RandomRotation(degrees=30),
                                             transforms.ToTensor(),
                                             normTransform
                                             ])

        testTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                            transforms.CenterCrop(crop_size),
                                            transforms.ToTensor(),
                                            normTransform
                                            ])

        meta_dir = os.path.join(meta_root, "SCSShip")
        data_dir = os.path.join(data_root, "SCSShip/SCSShip")
        query_dataset = SCSShip(meta_dir, os.path.join(data_dir, "test"), "query", testTransform,
                                meta_file=meta_file)
        gallery_dataset = SCSShip(meta_dir, os.path.join(data_dir, "train"), "gallery", testTransform,
                                  meta_file=meta_file)
        train_gallery_dataset = SCSShip(meta_dir, os.path.join(data_dir, "train"), "gallery",
                                        trainTransform,
                                        meta_file=meta_file)
        db_dataset = SCSShip(meta_dir, os.path.join(data_dir, "train"), "db", trainTransform,
                             meta_file=meta_file)
    else:
        print('no dataset avail')
        exit(0)

    args.meta_dir = meta_dir

    if query_dataset is not None:
        query_loader = DataLoader(query_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    if gallery_dataset is not None:
        gallery_loader = DataLoader(gallery_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    if db_dataset is not None:
        db_loader = DataLoader(db_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True, pin_memory=True)

    if dataset in ["FGSC-23", "SCSShip"]:
        train_gallery_loader = DataLoader(train_gallery_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True, pin_memory=True)
        return query_loader, gallery_loader, db_loader, train_gallery_loader

    return query_loader, gallery_loader, db_loader
