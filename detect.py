import argparse
import time

import torch

from configs import get_cfg
from models.ssd import SSD
from models.centernet import CenterNet
from predict import ssd_decode, centernet_decode

WEIGHTS = "saves/CenterNet_voc_epoch_200.pth"
IMAGE_PATHS = ["test/2007_000032.jpg", "test/2007_000033.jpg",
               "test/2007_000039.jpg"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help="experiment configure file name")
    args = parser.parse_args()
    return args


def main():
    t0 = time.time()
    args = parse_args()
    cfg, model_name = get_cfg(args.cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "ssd":
        model = SSD(cfg).to(device)
        model.load_state_dict(torch.load(WEIGHTS, map_location=device)["model"])
        print(f"Loaded weights: {WEIGHTS}")
        for img in IMAGE_PATHS:
            ssd_decode.detect_one_image(cfg, model, img, print_on=True, save_result=True, device=device)
    elif model_name == "centernet":
        model = CenterNet(cfg).to(device)
        model.load_state_dict(torch.load(WEIGHTS, map_location=device)["model"])
        print(f"Loaded weights: {WEIGHTS}")
        for img in IMAGE_PATHS:
            centernet_decode.detect_one_image(cfg, model, img, print_on=True, save_result=True, device=device)

    print(f"Total time: {(time.time() - t0):.2f}s")


if __name__ == '__main__':
    main()
