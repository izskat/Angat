import os
import torch
import time
import datetime
import numpy as np
import json
import cv2
import random
import matplotlib.pyplot as plt

from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.config import get_cfg
import detectron2.config
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# imports for validation loss hook
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import logging

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)

class MyTrainer(DefaultTrainer):
    @classmethod
#   Defines the method for model evaluation
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

#   Adds model hooks
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks
    
#Check if CUDA is available, and how many
if torch.cuda.is_available():
    print(f'CUDAs available: {torch.cuda.device_count()}')
    torch.cuda.set_device(2) #Set default GPU here
else:
    print("CUDA is not available")

# Change this for the source folder 
rootFolder = './Dataset'

# Train
trainDataDir = rootFolder + '/augmented_with_rotations'
trainAnnFile = rootFolder + '/augmented_with_rotations/augmented_train.json'
trainRoofAnnFile =  rootFolder + '/augmented_with_rotations/roof_train.json'
trainSsAnnFile =  rootFolder + '/augmented_with_rotations/ss_train.json'

# Test
testDataDir = rootFolder + '/test/images'
testAnnFile = rootFolder + '/test/test.json'
testRoofAnnFile =  rootFolder + '/test/roof_test.json'
testSsAnnFile =  rootFolder + '/test/ss_test.json'

# Val
valDataDir = rootFolder + '/val/images'
valAnnFile = rootFolder + '/val/val.json'
valRoofAnnFile =  rootFolder + '/val/roof_val.json'
valSsAnnFile =  rootFolder + '/val/ss_val.json'

# Register the dataset for it to be used by Detectron 2
register_coco_instances("train_dataset", {}, trainAnnFile, trainDataDir)
train_dataset = DatasetCatalog.get("train_dataset")

register_coco_instances("val_dataset", {}, valAnnFile, valDataDir)
val_dataset = DatasetCatalog.get("val_dataset")

'''
Hyper parameters to test:
- Backbone
- Learning Rate
- Max Iterations
'''
backbones = [101]
learning_rates = [0.01]  # Based on the first 2 runs above, we’ve decided to use more aggressive LR values
max_iters = [50000]
filename = 'Rooftop Identification Model'

for backbone in backbones:
    for learning_rate in learning_rates:
        for max_iter in max_iters:
            output_dir = f'./output_models/{filename}'
            os.makedirs(output_dir, exist_ok=True)

            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_" + str(backbone) + "_FPN_3x.yaml"))
            cfg.DATASETS.TRAIN = ("train_dataset",)
            # Let training initialize from model zoo
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_" + str(backbone) + "_FPN_3x.yaml")  
            cfg.SOLVER.IMS_PER_BATCH = 16
            cfg.SOLVER.BASE_LR = learning_rate
            cfg.SOLVER.MAX_ITER = max_iter
            '''
            CHANGE DECAY STEPS BASED ON DATASET SIZE!!!
            Total images / Batch size * 10 (epochs) = solver steps 
            4485 / 16 * 10 = 2800~

            '''
            cfg.SOLVER.STEPS = [240 * x for x in range(1, (max_iter//240) + 1)]        # learning rate decay
            cfg.SOLVER.GAMMA = 0.5
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 19  
            cfg.OUTPUT_DIR = output_dir

            # Other Configs
            cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
            cfg.INPUT.RANDOM_FLIP = "none"
            cfg.TEST.DETECTIONS_PER_IMAGE = 200
            cfg.SOLVER.CHECKPOINT_PERIOD = 1000

            # Validation Configs
            cfg.DATASETS.TEST = ("val_dataset",)
            cfg.TEST.EVAL_PERIOD = 2000

            trainer = MyTrainer(cfg)
            trainer.resume_or_load(resume=True)
            trainer.train()
