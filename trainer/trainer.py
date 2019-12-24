import os
import re
import abc
import time
import logging
import shutil
import torch
import numpy as np

from tqdm import tqdm
from torch import nn, optim
from torchreid.metrics.accuracy import accuracy
from torchreid.metrics.rank import evaluate_rank
from tensorboardX import SummaryWriter

from model import DIMNModel
from data_loader import build_data_loader
from .loss import RegularizationLoss, TripletLoss

torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

logger = logging.getLogger(__name__)



class Trainer:
    def __init__(self, config, mode, use_cpu):
        super().__init__()

        self.config = config
        self.mode_config = config["modes"][mode]
        self.mode = mode
        self.use_cpu = use_cpu
        self.device = None
        self.report_step = None
        self.save_step = None
        self.global_epoch = 1
        self.global_step = 1

        # attrs need to build
        self.pid_num = None
        self.train_data_loader = None
        self.eval_data_loader = None
        self.domain_dataset = None  # Only for eval mode
        self.model = None
        self.optim = None
        self.loss = None
        self.writer = None

        
    def init_attributes(self):
        use_gpu = not self.use_cpu and torch.cuda.is_available()
        self.use_cpu = not use_gpu
        self.device = 'cuda:{}'.format(self.config["cuda.id"]) if use_gpu else 'cpu'
        if self.mode in ["train", "resume"]:
            self.report_step = self.mode_config["report_step"]
            self.save_step = self.mode_config["save_step"]


    def build_model(self):
        self.model = DIMNModel(self.pid_num, self.config["model.params"])
        self.model.to(self.device)


    def build_train_data_loader(self):
        self.train_data_loader, self.pid_num = build_data_loader(
            data_dir=self.mode_config["data_dir"],
            domain_datasets=self.mode_config["domain_datasets.source"],
            batch_id_size=self.mode_config["train_batch_id_size"],
            num_workers=self.mode_config["train_num_workers"],
            mode="train",
            num_instances=2
        )

    
    def build_eval_data_loader(self, domain_dataset):
        self.domain_dataset = domain_dataset
        self.eval_data_loader, self.pid_num = build_data_loader(
            data_dir=self.mode_config["data_dir"],
            domain_datasets=domain_dataset,
            batch_id_size=self.mode_config["eval_batch_size"],
            num_workers=self.mode_config["eval_num_workers"],
            mode="eval"
        )


    def build_optim(self):
        # NOTE: Must init optimizer after the model is moved to expected device to ensure the
        # consistency of the optimizer state dtype
        lr, gamma = self.mode_config["lr"], self.mode_config["gamma"]
        self.optim = optim.Adam(self.model.parameters(), lr=lr)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(
            self.optim,
            gamma=gamma,
            last_epoch=-1
        )


    def build_loss(self):
        params = self.config["loss"]
        self.id_loss = nn.CrossEntropyLoss(reduction="mean").to(self.device)
        self.mat_loss = nn.CrossEntropyLoss(reduction="mean").to(self.device)
        self.reg_loss = RegularizationLoss(reduction="mean").to(self.device)
        self.tri_loss = TripletLoss(delta=params["tri_loss.delta"], reduction="mean").to(self.device)

    
    def build_summary_writer(self):
        save_dir = os.path.join(self.config["checkpoints.save_dir"], "events")
        self.writer = SummaryWriter(save_dir)


    def build(self, domain_dataset=None):
        if self.mode == 'train':
            self.init_attributes()
            self.build_train_data_loader()
            self.build_model()
            self.build_optim()
            self.build_loss()
            self.build_summary_writer()

        elif self.mode == 'resume':
            self.init_attributes()
            self.build_train_data_loader()
            self.build_model()
            self.build_optim()
            self.build_loss()
            self.load_checkpoint()
            self.build_summary_writer()

        elif self.mode == 'eval':
            self.init_attributes()
            self.build_eval_data_loader(domain_dataset)
            self.build_model()
            self.load_checkpoint()
            self.build_summary_writer()
            
        elif self.mode == "quantization":
            pass

        else:
            raise ValueError('Wrong mode \'{}\' is given.'.format(mode))


    def train_run(self, epoch, total_epoch):
        self.model.train()

        tqdm_ = tqdm(
            self.train_data_loader, unit="it",
            bar_format="{l_bar}{{bar}}{r_bar}".format(
                l_bar="{desc}{percentage:3.0f}%|",
                r_bar="| {rate_fmt} [{n_fmt}/{total_fmt}{postfix}]"
            )
        )
        
        with tqdm_ as pbar:
            total_time = 0
            total_loss, total_id_loss, total_mat_loss, total_reg_loss, total_tri_loss = 0, 0, 0, 0, 0
            total_step = 0

            pbar.set_description_str("|g-epoch: {}|".format(self.global_epoch))

            for images, cls_, _, fn in pbar:
                step_time_start = time.time()

                images = images.view(-1, 2, 3, 224, 224)
                images_probe, images_gallery = images[:, 0].squeeze(1), images[:, 1].squeeze(1)
                images_probe, images_gallery = images_probe.to(self.device), images_gallery.to(self.device)
                cls_ = cls_.view(-1, 2)[:, 0].to(self.device)
                
                features_probe, logits_classifier, logits_mapping_network, pred_cls_weights, selected_cls_weights, tmp_memory =\
                     self.model(images_probe, images_gallery, cls_)
                id_loss = self.id_loss(logits_classifier, cls_)
                mat_loss = self.mat_loss(logits_mapping_network, cls_)
                reg_loss = self.reg_loss(pred_cls_weights, selected_cls_weights)
                tri_loss = self.tri_loss(logits_mapping_network, cls_)
                loss = id_loss * self.config["loss.id_loss.weight"] + mat_loss * self.config["loss.mat_loss.weight"] +\
                    reg_loss * self.config["loss.reg_loss.weight"] + tri_loss * self.config["loss.tri_loss.weight"]
                loss.backward()
                
                self.optim.step()

                # stats
                step_time = time.time() - step_time_start
                step_time_start = time.time()
                total_time += step_time
                total_loss += loss.cpu().detach().item()
                total_id_loss += id_loss.cpu().detach().item()
                total_mat_loss += mat_loss.cpu().detach().item()
                total_reg_loss += reg_loss.cpu().detach().item()
                total_tri_loss += tri_loss.cpu().detach().item()

                pbar.set_postfix_str("loss: {:.5f}, step time: {:.2f}, g-step: {}".format(loss.cpu().detach().item(), step_time, self.global_step))
                
                self.optim.zero_grad()
                self.global_step += 1
                total_step += 1

                if self.global_step % self.report_step == 0:
                    avg_loss = total_loss / total_step
                    avg_id_loss, avg_mat_loss, avg_reg_loss, avg_tri_loss =\
                         total_id_loss / total_step, total_mat_loss / total_step, total_reg_loss / total_step, total_tri_loss / total_step
                    logger.info('| Average Step Time: {:.2f} s'.format(total_time / total_step))
                    logger.info('| Average Step Loss: {:.5f}'.format(avg_loss))
                    logger.info('| Average id/mat/reg/tri Losses: {:.5f}/{:.5f}/{:.5f}/{:.5f}'.\
                        format(avg_id_loss, avg_mat_loss, avg_reg_loss, avg_tri_loss))
                    
                    self.writer.add_scalars(
                        "loss",
                        {
                            "avg_loss": avg_loss,
                            "avg_id_loss": avg_id_loss,
                            "avg_mat_loss": avg_mat_loss,
                            "avg_reg_loss": avg_reg_loss,
                            "avg_tri_loss": avg_tri_loss
                        },
                        self.global_step
                    )
                    self.writer.add_scalar("lr", self.optim.param_groups[0]["lr"], self.global_step)
                    self.writer.flush()
                    total_time, total_step, total_loss, total_id_loss, total_mat_loss, total_reg_loss, total_tri_loss = 0, 0, 0, 0, 0, 0, 0
                
                if self.global_step % self.save_step == 0:
                    self.save_model()

                # if not self.use_cpu and torch.cuda.memory_cached() > self.MAX_CUDA_MEMORY_CACHE:
                #     torch.cuda.empty_cache()

        self.global_epoch += 1                
        self.lr_scheduler.step()

        # save trained model at the end of one epoch
        self.save_model()


    def eval_run(self):
        self.model.eval()

        total_cmc_top1, total_cmc_top5, total_cmc_top10, total_ap, total_count = 0, 0, 0, 0, 0

        for probe, gallery in self.eval_data_loader():
            cls_list = []
            gallery_cid_list = []
            with tqdm(gallery) as pbar:
                for images, cls_, cids, _ in pbar:
                    for c, cid in zip(cls_, cids):
                        if c.item() not in cls_list:
                            cls_list.append(c.item())
                            gallery_cid_list.append(cid.item())
                    new_cls_ = torch.tensor([cls_list.index(c) for c in cls_], dtype=torch.long)
                    images, new_cls_ = images.to(self.device), new_cls_.to(self.device)
                    self.model.inference_gallery(images, new_cls_)
            
            pred_logit_list, target_cls_list = [], []
            probe_cid_list = []
            with tqdm(probe) as pbar:
                for images, cls_, cids, _ in pbar:
                    new_cls_ = torch.tensor([cls_list.index(c.item()) for c in cls_], dtype=torch.long)
                    images, new_cls_ = images.to(self.device), new_cls_.to(self.device)
                    pred_logits = self.model.inference_probe(images)
                    probe_cid_list += [c.item() for c in cids]
                    pred_logit_list.append(pred_logits)
                    target_cls_list.append(new_cls_)

            pred_logit = torch.cat(pred_logit_list, dim=0).view(-1, pred_logits.size(1))
            pred_logit = torch.nn.functional.softmax(pred_logit, dim=1)
            pred_logit = 1 / (pred_logit + 1e-7)
            
            target_cls = torch.cat(target_cls_list, dim=0).view(-1)

            cmc, ap = evaluate_rank(
                np.array(pred_logit.detach().cpu()), np.array(target_cls.detach().cpu()), np.array(range(len(cls_list))), np.array(probe_cid_list), np.array(gallery_cid_list),
                max_rank=10, 
                use_cython=True
            )
            
            total_cmc_top1 += cmc[0]
            total_cmc_top5 += cmc[4]
            total_cmc_top10 += cmc[9]
            total_ap += ap
            total_count += 1
        
        top1 = total_cmc_top1 / total_count
        top5 = total_cmc_top5 / total_count
        top10 = total_cmc_top10 / total_count
        mAP = total_ap / total_count

        logging.info("| Metric of {}: top1/top5/top10/mAP: {:.4f}/{:.4f}/{:.4f}/{:.4f}".format(self.domain_dataset, top1, top5, top10, mAP))
        self.writer.add_scalars(
            "metric",
            {
                "{}_top1".format(self.domain_dataset): top1,
                "{}_top5".format(self.domain_dataset): top5,
                "{}_top10".format(self.domain_dataset): top10,
                "{}_mAP".format(self.domain_dataset): mAP,
            },
            self.global_step
        )
        self.writer.flush()


    def save_model(self):
        save_dir = self.config["checkpoints.save_dir"]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.config["modes"][self.mode]["timestamp"]:
            save_path = os.path.join(save_dir, 'model_{}_{}.pkl'.format(self.model.__class__.__name__, str(int(time.time()))))
        else:
            save_path = os.path.join(save_dir, 'model_{}.pkl'.format(self.model.__class__.__name__))

        # Dump the state_dict of model in cpu mode
        model_state_dict = self.model.cpu().state_dict()

        # NOTE: do remember to move model back to right device because we just called self.model.cpu() above
        self.model.to(self.device)

        # Save checkpoint
        torch.save({
            "model_state_dict": model_state_dict,
            "optim_state_dict": self.optim.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "global_epoch": self.global_epoch,
            "global_step": self.global_step,
            },
            save_path
        )

        logger.info('| Checkpoint is saved successfully in \'{}\''.format(save_path))


    def load_checkpoint(self):
        if self.mode in ["train", "resume"]:
            logger.info("| Load checkpoint from {} ...".format(self.config["checkpoints.load_path"]))

            checkpoint = torch.load(self.config["checkpoints.load_path"])

            # load model state_dict
            self.model.load_state_dict(checkpoint["model_state_dict"])

            # load optim state_dict
            self.optim.load_state_dict(checkpoint["optim_state_dict"])

            # load scheduler state_dict
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

            # load global step
            self.global_step = checkpoint["global_step"]
            self.global_epoch = checkpoint["global_epoch"]

            logger.info('| Model is loaded successfully from \'{}\''.format(self.config["checkpoints.load_path"]))
        
        elif self.mode == "eval":
            logger.info("| Load checkpoint from {} ...".format(self.config["checkpoints.load_path"]))

            checkpoint = torch.load(self.config["checkpoints.load_path"])
            loaded_model_state_dict = checkpoint["model_state_dict"]

            state_dict = self.model.state_dict()
            # FIXME
            for name, param in loaded_model_state_dict.items():
                if re.match(r"memory_bank*|classifier*", name) is None:
                    state_dict[name].copy_(param)
            
            # load global step
            self.global_step = checkpoint["global_step"]
            self.global_epoch = checkpoint["global_epoch"]
                    