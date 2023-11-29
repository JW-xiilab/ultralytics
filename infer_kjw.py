"""by lyuwenyu
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '4'
import sys
import time
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from copy import deepcopy
from loguru import logger
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.augment import LetterBox
from ultralytics.engine.model_kjw import Model as YOLO
from ultralytics.engine.results import Results
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import ops, callbacks, DEFAULT_CFG
from ultralytics.utils.torch_utils import select_device


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/root/dataset/coco2017/images/val2017', type=str,)
    # parser.add_argument('--config', '-c', default='ultralytics/cfg/default.yaml', type=str, )
    parser.add_argument('--model', default='models/yolov8x.pt', type=str, )
    parser.add_argument('--save_dir', default='YOLOv8_outputs_g_noised/')
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="device to run the model on demand, can either be cpu or gpu. It decides which device to run automatically",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.1,
        help="the spread or amplitude of the Gaussian noise distribution",
    )
    parser.add_argument(
        "--save_txt",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument(
        "--save_conf",
        action="store_false",
        help="whether to save the predicted confidence score of detected object",
    )
    parser.add_argument('--classes', nargs='*', default=[0], type=int)

    return parser


def get_model_info(model: nn.Module, tsize: Sequence[int]) -> str:
    from thop import profile

    stride = 640  # 64
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    return info


class Predictor():

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        # self.args = {**self.args, **args, 'mode':'predict'}
        # self.save_dir = get_save_dir(self.args)
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        self.done_warmup = False

        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.plotted_img = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.results = None
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.txt_path = None
        callbacks.add_integration_callbacks(self)

    def preprocess(self, im):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

    def pre_transform(self, im):
        """
        Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Returns:
            (list): A list of transformed images.
        """
        same_shapes = all(x.shape == im[0].shape for x in im)
        letterbox = LetterBox(auto=same_shapes and self.model.pt, stride=self.model.stride)
        return [letterbox(image=x) for x in im]

    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        self.model = AutoBackend(model or self.args.model,
                                 device=select_device(self.args.device, verbose=verbose),
                                 dnn=self.args.dnn,
                                 data=self.args.data,
                                 fp16=self.args.half,
                                 fuse=True,
                                 verbose=verbose)

        self.device = self.model.device  # update device
        self.args.half = self.model.fp16  # update half
        self.model.eval()
        logger.info("Model Summary: {}".format(get_model_info(self.model, (640,640))))

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            pred = torch.cat((pred[:,-1:], pred[:, :-1]), dim=1)
            results.append(pred)
            # img_path = self.batch[0][i]
            # results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        if len(results) != 1:
            print(1)
            raise
        return results[0]


class Model(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        # self.device = args.device
        logger.info("Loading model")
        self.model = YOLO(args.model)
        self.noise = self.args.__dict__.pop('noise_std', None)
        self.args.mode = 'predict'
        self.predictor = Predictor(overrides=vars(self.args), _callbacks=self.model.callbacks)
        self.predictor.setup_model(model=self.model.model, verbose=False)
        self.device = self.predictor.device
        # self.model = dist.warp_model(cfg.model.to(self.device), cfg.find_unused_parameters, cfg.sync_bn)
        # self.criterion = args.criterion.to(self.device)
        self.postprocessor = self.predictor.postprocess
        # self.postprocessor = PostProc(classes=cfg.classes, iou_thres=cfg.iou_thres)#cfg.postprocessor
        self.model.model.eval()         
        logger.info("Model loaded")
        # self.test_size = (640, 640)
        # logger.info("Model Summary: {}".format(get_model_info(self.model, self.test_size)))
        # self.transform = T.Compose([T.Resize(self.test_size),#self.test_size),
        #                             T.ToImageTensor(),
        #                             T.ConvertDtype()])
        

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
            # img = Image.open(img).convert('RGB')
        else:
            img_info["file_name"] = None
            return None, None
 
        # height, width = img.shape[:2]
        # # width, height = img.size
        # img_info["height"] = height
        # img_info["width"] = width
        # img_info["raw_img"] = img

        t_img = self.predictor.preprocess([img])
        if self.noise:
            noise = torch.randn_like(t_img) * self.noise
            t_img = t_img + noise
            t_img = torch.clamp(t_img, 0, 1)

        # img = img.unsqueeze(0)
        # orig_target_size = torch.tensor([width, height])
        if self.device.type == 'cuda':
            t_img = t_img.cuda()
            # orig_target_size = orig_target_size.cuda()

        with torch.no_grad():
            outputs = self.predictor.model(t_img)
        results = self.postprocessor(outputs, t_img, [img])
        
        return results, img_info

def main(args, ) -> None:
    '''main
    '''

    # if args.resume:
    #     checkpoint = torch.load(args.resume, map_location='cpu') 
    #     if 'ema' in checkpoint:
    #         state = checkpoint['ema']['module']
    #     else:
    #         state = checkpoint['model']
    # else:
    #     raise AttributeError('only support resume to load model.state_dict by now.')

    # args.model.load_state_dict(state)
    # if args.classes:
    #     args.classes = args.classes
    # if args.iou_thres:
    #     args.iou_thres = args.iou_thres

    model = Model(args)
    current_time = time.localtime()
    # args.out_dir = os.path.join(args.out_dir, str(args.classes[0]))
    image_demo(model, args.save_dir, args.source, current_time, args.classes, save_txt=args.save_txt)

def image_demo(model, out_dir, path, current_time, classes, save_result=None, save_txt=None):
    out_dir = Path(__file__).parents[-5].absolute() / out_dir 
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    img_dir = Path(path)
    if img_dir.is_dir():
        files = get_image_list(path)
    elif img_dir.is_file(path):
        files = [path]
    else:
        print("The given directory/path doesn't seem to exist")
        raise
    files.sort()
    cnt = 0
    if save_txt:
        txt_folder = os.path.join(out_dir, 'labels', str(classes[0]))
        save_folder = os.path.join(
            txt_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        # vis_folder = os.path.join(save_folder, 'images')
        # save_folder = os.path.join(
        #     vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        # )
        
        os.makedirs(save_folder,  exist_ok=True)
    for image_name in tqdm(files, desc='Inferencing', total=len(files), leave=True):
        p = Path(image_name)
        outputs, _ = model.inference(image_name)
        if not isinstance(outputs, torch.Tensor) or len(outputs) == 0:
            continue
        cnt += 1
        if save_txt:
            txt_file = os.path.join(save_folder, p.stem + '.txt')
            save_txt_file(txt_file, outputs) 
    logger.info(f'Results saved to {Path(txt_folder).resolve()}')
    logger.info(f'{cnt} labels saved to {Path(save_folder).resolve()}')

def save_txt_file(txt_file, result):
    texts = []
    for d in result:
        c, conf = int(d[0]), float(d[-1])
        line = (c, *d[1:-1].view(-1), conf)
        texts.append(('%g ' * len(line)).rstrip() % line)
    if texts:
        Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
        with open(txt_file, 'a') as f:
            f.writelines(text + '\n' for text in texts)

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


if __name__ == '__main__':

    args = make_parser().parse_args()

    main(args)
