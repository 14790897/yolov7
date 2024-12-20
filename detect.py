import argparse
import time
from pathlib import Path
import os, json
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size,
    check_requirements,
    check_imshow,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
)
from utils.plots import plot_one_box
from utils.torch_utils import (
    select_device,
    load_classifier,
    time_synchronized,
    TracedModel,
)


def detect(opt, save_img=False):
    source, weights, view_img, save_txt, imgsz, trace, save_json = (
        opt.source,
        opt.weights,
        opt.view_img,
        opt.save_txt,
        opt.img_size,
        not opt.no_trace,
        opt.save_json,
    )
    save_img = not opt.nosave and not source.endswith(".txt")  # save inference images
    webcam = (
        source.isnumeric()
        or source.endswith(".txt")
        or source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    )

    # Directories
    save_dir = Path(
        increment_path(Path(opt.project), exist_ok=opt.exist_ok)
    )  # increment run
    (save_dir / "labels" if save_txt or save_json else save_dir).mkdir(
        parents=True, exist_ok=True
    )  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name="resnet101", n=2)  # initialize
        modelc.load_state_dict(
            torch.load("weights/resnet101.pt", map_location=device)["model"]
        ).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != "cpu":
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))
        )  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != "cpu" and (
            old_img_b != img.shape[0]
            or old_img_h != img.shape[2]
            or old_img_w != img.shape[3]
        ):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            classes=opt.classes,
            agnostic=opt.agnostic_nms,
        )
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], "%g: " % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / "labels" / p.stem) + (
                "" if dataset.mode == "image" else f"_{frame}"
            )  # img.txt
            json_path = str(save_dir / "output")
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (
                            (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                            .view(-1)
                            .tolist()
                        )  # normalized xywh
                        line = (
                            (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        )  # label format
                        with open(txt_path + ".txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")
                    if save_json:  # Write to JSON file
                        # 将 xyxy 转换为列表形式
                        xyxy_list = [float(x) for x in xyxy]

                        # 创建一个字典来保存当前的检测结果
                        detection_data = {
                            "xyxy": xyxy_list,  # 边界框的坐标
                            "cls": int(cls),  # 检测到的类别
                            "confidence": float(conf),  # 检测置信度
                            "file_path": str(p),  # 图像文件路径
                        }

                        # 将数据追加到 JSON 文件
                        json_file_path = json_path + ".json"
                        if Path(json_file_path).exists():
                            with open(json_file_path, "r+") as f:
                                existing_data = json.load(f)
                                existing_data.append(detection_data)
                                f.seek(0)
                                json.dump(existing_data, f, indent=4)
                        else:
                            with open(json_file_path, "w") as f:
                                json.dump([detection_data], f, indent=4)

                    if save_img or view_img:  # Add bbox to image
                        label = f"{names[int(cls)]} {conf:.2f}"
                        plot_one_box(
                            xyxy,
                            im0,
                            label=label,
                            color=colors[int(cls)],
                            line_thickness=1,
                        )

            # Print time (inference + NMS)
            print(
                f"{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS"
            )

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += ".mp4"
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                        )
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = (
            f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
            if save_txt
            else ""
        )
        # print(f"Results saved to {save_dir}{s}")

    print(f"Done. ({time.time() - t0:.3f}s)")


# 读取 closest_point 并进行相应操作
def process_closest_point(closest_point, initial_result_directory):
    opt = parser.parse_args()
    id = closest_point["ID"]
    frame = closest_point["Frame"]
    y = (closest_point["Box"][1] + closest_point["Box"][3]) / 2
    # 计算 frame//2
    half_frame = frame // 2

    # 创建 frame_range 列表
    frame_range = list(range(half_frame - 5, half_frame + 6))
    id_folder_path = os.path.join(initial_result_directory, str(id))

    images_x_output_path = os.path.join(id_folder_path, "images_x_output")
    opt.project = images_x_output_path
    print("images_x_output_path:", images_x_output_path)

    print(
        f"ID: {id}, Frame: {frame}, Half Frame: {half_frame}, Frame Range: {frame_range}，纵坐标：{y}"
    )

    image_dir_path = "images_x"

    for number in frame_range:
        image_file_name = f"x-{number}.jpg"

        image_file_path = os.path.join(image_dir_path, image_file_name)

        opt.source = image_file_path
        # print(opt)
        with torch.no_grad():
            if opt.update:  # update all models (to fix SourceChangeWarning)
                for opt.weights in ["yolov7.pt"]:
                    detect(opt)
                    strip_optimizer(opt.weights)
            else:
                detect(opt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default="best_particle.pt",
        help="model.pt path(s)",
    )
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-json", action="store_true", help="save results to *.json", default=True
    )
    parser.add_argument(
        "--save-conf",
        action="store_true",
        help="save confidences in --save-txt labels",
    )
    parser.add_argument(
        "--nosave", action="store_true", help="do not save images/videos"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --class 0, or --class 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
        default=True,  # 设置默认值为 True
    )
    parser.add_argument("--no-trace", action="store_true", help="don`t trace model")
    initial_result_directory = "initial_result"
    stats_file_path = os.path.join(initial_result_directory, "all_stats.json")
    # 检查文件是否存在
    if os.path.isfile(stats_file_path):
        with open(stats_file_path, "r") as file:
            all_stats = json.load(file)
            for k, v in all_stats.items():
                # 读取并处理 closest_point
                if "closest_point" in v:
                    closest_point = v["closest_point"]
                    process_closest_point(closest_point, initial_result_directory)
                else:
                    print(f"No closest_point found in {stats_file_path}")
                    break
    else:
        print(f"File {stats_file_path} not found")
        exit()
