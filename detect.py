import os
import shutil
import time

import cv2
import torch
from tqdm import tqdm

from registry import model_registry
from utils.ckpt import CheckPoint

# 配置文件路径
CONFIG = "configs/yolo7_cfg.py"
# 权重文件位置
WEIGHTS = "saves/yolov7_weights.pth"
# 输入文件类型：视频还是图片
TYPE = "video"  # or "video"
# 测试图片路径的列表
IMAGE_PATHS = ["test/2007_002273.jpg"]
# 原视频路径
SRC_VIDEO = "test/1.flv"
# 目标视频路径
DST_VIDEO = "test/det_1.mp4"
ONLINE = True


def detect_video(model, src_video_path, dst_video_path, decode_fn, temp_frames='frames', online=False):
    # 创建临时文件夹，用于存储测试过程中产生的临时文件
    if not os.path.exists(temp_frames):
        os.makedirs(temp_frames)

    # 读取原始视频
    video_capture = cv2.VideoCapture(src_video_path)

    # 码率
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    # 帧尺寸
    frame_size = (
        int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(dst_video_path, fourcc, fps, frame_size)

    if online:
        # 在线检测：在读取视频的过程中直接检测，并显示fps
        while True:
            ret, frame = video_capture.read()
            if ret:
                t0 = time.time()
                cv2.imwrite(os.path.join(temp_frames, f"frame_temp.jpg"), frame)
                frame_dir = os.path.join(temp_frames, f"frame_temp.jpg")
                new_frame = decode_fn(model, frame_dir, print_on=False, save_result=False)
                t1 = time.time()
                fps = (1 / (t1-t0) + fps) / 2
                # 加上帧率显示
                new_frame = cv2.putText(new_frame, f"fps= {fps:.2f}", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.namedWindow(f"detection_results", 0)
                cv2.imshow(f"detection_results", new_frame)
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break

    else:
        # 离线检测：将视频分解成帧序列，对每一帧进行检测，然后合并检测后的结果为一个新的视频
        frame_counter = 0
        while True:
            ret, frame = video_capture.read()
            if ret:
                # 存储每一帧的图片
                cv2.imwrite(os.path.join(temp_frames, f"frame_{frame_counter}.jpg"), frame)
                if frame_counter % 50 == 0:
                    print(f"====================>保存第{frame_counter}帧<==================")
                frame_counter += 1
            else:
                break

        print(f"视频{src_video_path}有{frame_counter}帧")

        for i in tqdm(range(frame_counter)):
            frame_dir = os.path.join(temp_frames, f"frame_{i}.jpg")
            new_frame = decode_fn(model, frame_dir, print_on=False, save_result=False)
            video_writer.write(new_frame)

    video_capture.release()
    video_writer.release()
    cv2.destroyAllWindows()
    shutil.rmtree(temp_frames)


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = os.path.basename(CONFIG)
    try:
        model_cfg, model_class = model_registry[cfg]
    except KeyError:
        raise ValueError(f"找不到配置文件：{cfg}.")

    model_object = model_class(model_cfg, device)
    model, _ = model_object.build_model()
    model.to(device)

    # 加载模型权重
    CheckPoint.load(WEIGHTS, device, model, pure=True)
    print(f"Loaded weights: {WEIGHTS}")

    assert TYPE in ["video", "image"], f"不支持{TYPE}类型的文件作为输入"
    if TYPE == "video":
        detect_video(model, src_video_path=SRC_VIDEO, dst_video_path=DST_VIDEO, decode_fn=model_object.predict, online=ONLINE)
    else:
        for img in IMAGE_PATHS:
            model_object.predict(model, img, print_on=True, save_result=True)

    print(f"Total time: {(time.time() - t0):.2f}s")


if __name__ == '__main__':
    main()
