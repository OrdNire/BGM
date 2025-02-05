if __name__ == '__main__':
    from config_args import get_args_train_SIR_exp2
    import argparse
    import torch
    from models.SpaceIR.SIR_exp3 import build_SpaceIR

    args = get_args_train_SIR_exp2(argparse.ArgumentParser())

    src = torch.rand(8, 3, 224, 224)
    label_embedding = torch.rand(17, 768)
    model = build_SpaceIR(args, label_embedding)
    classifier_output, distance_output, global_embed = model(src, stage="classification")