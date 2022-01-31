import logging
import os, sys
import shutil
from datetime import datetime
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from volsim.metrics import *
from volsim.params import *

class Logger(object):
    def __init__(self, path:str, params:Params=None, override:bool=False, addNumber:bool=True, addDate:bool=False):
        if addDate:
            self.path = "runs/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + path
        elif addNumber:
            self.path = "runs/%s_%02d" % (path, 0)
        else:
            self.path = "runs/" + path

        if os.path.isdir(self.path):
            if override:
                shutil.rmtree(self.path)
            else:
                if addNumber:
                    num = 1
                    while os.path.isdir(self.path):
                        self.path = "runs/%s_%02d" % (path, num)
                        num += 1
                else:
                    raise ValueError("Model directory already exists!")

        os.makedirs(self.path)
        shutil.copy("src/training.py", os.path.join(self.path, "training.py"))

        self.tfWriter = CustomSummaryWriter(self.path, flush_secs=20)

        # hacky reload fix for logging to work properly
        import importlib
        importlib.reload(logging)
        logging.basicConfig(filename=self.path+"/log.txt", format="%(asctime)s %(message)s", level=logging.INFO, datefmt="%H:%M:%S")
        logging.info("Path: %s" % self.path)
        logging.info("PyTorch Seed: %d" % torch.random.initial_seed())
        if params:
            logging.info(str(params.asDict()))

    def setup(self, model:nn.Module, optimizer:Optimizer, lrScheduler:_LRScheduler, valSplit:dict, testSplit:dict):
        self.model = model
        self.optimizer = optimizer
        self.lrScheduler = lrScheduler

        datasetsCor = {}
        for split in valSplit:
            datasetsCor[split] = ["Multiline", ["datasets/Correlation_" + split]]
        for split in testSplit:
            datasetsCor[split] = ["Multiline", ["datasets/Correlation_" + split]]

        datasetsCor["All (Val)"] = ["Multiline", ["datasets/Correlation_ValAll"]]
        datasetsCor["All (Test)"] = ["Multiline", ["datasets/Correlation_TestAll"]]

        layout = {
            "Training":{
                "Correlation":          ["Multiline", ["train/Epoch_CorrelationFull"]],
                "Correlation (Mean)":   ["Margin",    ["train/Epoch_CorrelationMean", "train/Epoch_CorrelationMeanLow", "train/Epoch_CorrelationMeanHigh"]],
                "Loss":                 ["Multiline", ["train/Epoch_Loss", "train/Epoch_LossL2", "train/Epoch_LossCorr", "train/Epoch_LossSizeReg", "train/Epoch_LossSlConvReg"]],
            },

            "Training Batches":{
                "Loss (Batch)":                 ["Multiline", ["train/Batch_Loss", "train/Batch_LossL2", "train/Batch_LossCorr", "train/Batch_LossSlConvReg"]],
                "Correlation (Batch)":          ["Multiline", ["train/Batch_Correlation"]],
                "Correlation (Sample Sliced)":  ["Multiline", ["train/Sample_Correlation"]],
            },
            
            "Validation":{
                "Correlation":          ["Multiline", ["val/Epoch_CorrelationFull"]],
                "Correlation (Mean)":   ["Margin",    ["val/Epoch_CorrelationMean", "val/Epoch_CorrelationMeanLow", "val/Epoch_CorrelationMeanHigh"]],
                "Distance":             ["Margin",    ["val/Epoch_Distance", "val/Epoch_DistanceLow", "val/Epoch_DistanceHigh"]],
            },

            "Validation Batches":{
                "Correlation (Batch)":  ["Multiline", ["val/Batch_Correlation"]],
            },

            "Test":{
                "Correlation":          ["Multiline", ["test/Epoch_CorrelationFull"]],
                "Correlation (Mean)":   ["Margin",    ["test/Epoch_CorrelationMean", "test/Epoch_CorrelationMeanLow", "test/Epoch_CorrelationMeanHigh"]],
                "Distance":             ["Margin",    ["test/Epoch_Distance", "test/Epoch_DistanceLow", "test/Epoch_DistanceHigh"]],
            },

            "Test Batches":{
                "Correlation (Batch)":  ["Multiline", ["test/Batch_Correlation"]],
            },

            "Datasets": datasetsCor,
        }
        self.tfWriter.add_custom_scalars(layout)

    def close(self):
        logging.info("\nLog completed.")
        logging.shutdown()
        self.tfWriter.close()


    def saveTrainState(self, epoch:int, milestone:bool=False):
        assert (self.model), "No model to save, setup logger first!"

        saveDict = {
            "epoch" : epoch,
            "optimizer" : self.optimizer.state_dict,
            "lrScheduler" : self.lrScheduler.state_dict
        }
        torch.save(saveDict, self.path + "/TrainState.pth")

        if milestone:
            self.model.save(self.path + "/Epoch%02d.pth" % (epoch), override=True, noPrint=True)
        else:
            self.model.save(self.path + "/Model.pth", override=True, noPrint=True)


    def resumeTrainState(self, epoch:int):
        if epoch <= 0:
            return

        assert (self.model), "No model to load, setup logger first!"
        saveDict = torch.load(self.path + "/TrainState.pth")
        assert (saveDict["epoch"] == epoch), "Epoch mismatch when loading train state."

        self.model.resume(self.path + "Model.pth")
        self.optimizer.load_state_dict(saveDict["optimizer"])
        schedulerState = saveDict.get("lrScheduler", None)
        if schedulerState:
            self.lrScheduler.load_state_dict(schedulerState)


# Adjust hParam behavior of SummaryWriter to store results in a single folder
# Workaround from:
# https://github.com/pytorch/pytorch/issues/32651#issuecomment-643791116
class CustomSummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict):
        # remove all lists from hParam dict since only int, float, str, bool and torch.Tensor are possible
        for key, value in hparam_dict.items():
            if type(value) is list:
                valueStr = " ".join([str(elem) for elem in value])
                hparam_dict[key] = valueStr
            elif not type(value) in [int, float, str, bool, torch.Tensor]:
                hparam_dict[key] = " "

        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()

        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)