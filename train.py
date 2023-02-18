import torch
import argparse


from trainer import Yolo3Trainer, SSDTrainer, CenterNetTrainer
from configs import get_cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help="experiment configure file name")
    parser.add_argument('--mode', type=str, required=True, help="train, test or predict")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg, model_name = get_cfg(args.cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m = None
    if model_name == "ssd":
        m = SSDTrainer(cfg, device)
    elif model_name == "centernet":
        m = CenterNetTrainer(cfg, device)
    elif model_name == "yolov3":
        m = Yolo3Trainer(cfg, device)

    if args.mode == "train":
        m.train()
    elif args.mode == "test":
        m.evaluate(weights="saves/ssd_voc_final.pth")
    else:
        raise ValueError(f"Unsupported mode：{args.mode}")


if __name__ == '__main__':
    main()