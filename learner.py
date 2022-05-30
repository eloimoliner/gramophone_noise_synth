import numpy as np
import os
import re
import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from glob import glob

#from dataset import from_path as dataset_from_path
from getters import get_sde
import time
from inference import  GramophoneSampler
import utils


class Learner:
    def __init__(
        self, model_dir, model, train_set, optimizer, args
    ):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.model = model
        if args.restore:
            self.restore_from_checkpoint()

        self.ema_weights = [param.clone().detach()
                            for param in self.model.parameters()]
        self.sde = get_sde(args.sde_type, args.sde_kwargs)

        self.sampler=GramophoneSampler(self.model,self.sde)

        self.ema_rate = args.ema_rate
        self.train_set = train_set
        self.optimizer = optimizer
        self.args = args
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.args.scheduler_step_size, gamma=self.args.scheduler_gamma)

        self.step = 0
        self.is_master = True

        self.loss_fn = nn.MSELoss()
        self.v_loss = nn.MSELoss(reduction="none")
        self.summary_writer = None
        self.n_bins = args.n_bins
        self.num_elems_in_bins_train = np.zeros(self.n_bins)
        self.sum_loss_in_bins_train = np.zeros(self.n_bins)
        self.num_elems_in_bins_test = np.zeros(self.n_bins)
        self.sum_loss_in_bins_test = np.zeros(self.n_bins)
        self.cum_grad_norms = 0

    def state_dict(self):
        if hasattr(self.model, "module") and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            "step": self.step,
            "model": {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in model_state.items()
            },
            'ema_weights': self.ema_weights,
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, "module") and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict["model"])
        else:
            self.model.load_state_dict(state_dict["model"])
        self.step = state_dict["step"]
        self.ema_weights = state_dict['ema_weights']

    def save_to_checkpoint(self, filename="weights"):
        save_basename = f"{filename}-{self.step}.pt"
        save_name = f"{self.model_dir}/{save_basename}"
        torch.save(self.state_dict(), save_name)

    def restore_from_checkpoint(self, checkpoint_id=None):
        try:
            if checkpoint_id is None:
                # find latest checkpoint_id
                list_weights = glob(f'{self.model_dir}/weights-*')
                id_regex = re.compile('weights-(\d*)')
                list_ids = [int(id_regex.search(weight_path).groups()[0])
                            for weight_path in list_weights]
                checkpoint_id = max(list_ids)

            checkpoint = torch.load(
                f"{self.model_dir}/weights-{checkpoint_id}.pt")
            self.load_state_dict(checkpoint)
            return True
        except (FileNotFoundError, ValueError):
            return False

    def sample(self):
        device = next(self.model.parameters()).device
        #noise = torch.randn(8, self.args.audio_len, device=device)
        res=self.sampler.predict_unconditional(100,1)
        res=res.flatten()
        return res


    def train(self):
        device = next(self.model.parameters()).device
        while True:
            start=time.time()
            
            y=self.train_set.next()    
            
            y=y.to(device)

            #features = _nested_map(
            #    features,
            #    lambda x: x.to(device) if isinstance(
            #        x, torch.Tensor) else x,
            #)
            loss, vectorial_loss, t= self.train_step(y)

            if torch.isnan(loss).any():
                raise RuntimeError(
                    f"Detected NaN loss at step {self.step}.")

            if self.step % self.args.log_interval == 0:
                t_detach = t.clone().detach().cpu().numpy()
                t_detach = np.reshape(t_detach, -1)
                vectorial_loss = torch.mean(vectorial_loss, 1).cpu().numpy()
                vectorial_loss = np.reshape(vectorial_loss, -1)
                self.update_conditioned_loss(vectorial_loss, t_detach, True)
                self._write_summary(self.step)

            if self.step % self.args.save_interval==0:
                #self.test_set_evaluation()

                #self._write_test_summary(self.step)
                #sample and log sample results, consider also the evaluation this guy is doing
                res=self.sample()
                
                self._write_summary_test(res,self.step)

                self.save_to_checkpoint()

            self.step += 1
            end=time.time()
            print("Step: ",self.step,", Loss: ",loss.item(),", Time: ",end-start)

    def train_step(self,y):
        for param in self.model.parameters():
            param.grad = None

        audio = y

        N, T = audio.shape
        device=audio.device

        t = torch.rand(N, 1, device=audio.device)
        t = (self.sde.t_max - self.sde.t_min) * t + self.sde.t_min

        noise = torch.randn_like(audio)
        noisy_audio = self.sde.perturb(audio, t, noise)
        sigma = self.sde.sigma(t)
        predicted = self.model(noisy_audio, sigma)
        loss = self.loss_fn(noise, predicted)

        loss.backward()
        self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        if self.is_master:
            self.update_ema_weights()


        vectorial_loss = self.v_loss(noise, predicted).detach()


        self.cum_grad_norms += self.grad_norm

        return loss, vectorial_loss, t

    def _write_summary_test(self,res, step):
        print("writing summary")
        writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
        writer.add_audio('val/audio', res, step, sample_rate=self.args.sample_rate)
        writer.add_image('val/spec', utils.generate_images(res), step, dataformats='WHC')
        writer.flush()
        self.summary_writer = writer
    def _write_summary(self, step):
        loss_in_bins_train = np.divide(
            self.sum_loss_in_bins_train, self.num_elems_in_bins_train
        )
        dic_loss_train = {}
        for k in range(self.n_bins):
            dic_loss_train["loss_bin_" + str(k)] = loss_in_bins_train[k]

        sum_loss_n_steps = np.sum(self.sum_loss_in_bins_train)
        mean_grad_norms = self.cum_grad_norms / self.num_elems_in_bins_train.sum() * \
            self.args.batch_size
        writer = self.summary_writer or SummaryWriter(
            self.model_dir, purge_step=step)

        writer.add_scalar('train/sum_loss_on_n_steps',
                          sum_loss_n_steps, step)
        writer.add_scalar("train/mean_grad_norm", mean_grad_norms, step)
        writer.add_scalars("train/conditioned_loss", dic_loss_train, step)
        writer.flush()
        self.summary_writer = writer
        self.num_elems_in_bins_train = np.zeros(self.n_bins)
        self.sum_loss_in_bins_train = np.zeros(self.n_bins)
        self.cum_grad_norms = 0

    def _write_test_summary(self, step):
        # Same thing for test set
        loss_in_bins_test = np.divide(
            self.sum_loss_in_bins_test, self.num_elems_in_bins_test
        )
        dic_loss_test = {}
        for k in range(self.n_bins):
            dic_loss_test["loss_bin_" + str(k)] = loss_in_bins_test[k]

        writer = self.summary_writer or SummaryWriter(
            self.model_dir, purge_step=step)
        writer.add_scalars("test/conditioned_loss", dic_loss_test, step)
        writer.flush()
        self.summary_writer = writer
        self.num_elems_in_bins_test = np.zeros(self.n_bins)
        self.sum_loss_in_bins_test = np.zeros(self.n_bins)

    def update_conditioned_loss(self, vectorial_loss, continuous_array, isTrain):
        continuous_array = np.trunc(self.n_bins * continuous_array)
        continuous_array = continuous_array.astype(int)
        if isTrain:
            for k in range(len(continuous_array)):
                self.num_elems_in_bins_train[continuous_array[k]] += 1
                self.sum_loss_in_bins_train[continuous_array[k]
                                            ] += vectorial_loss[k]
        else:
            for k in range(len(continuous_array)):
                self.num_elems_in_bins_test[continuous_array[k]] += 1
                self.sum_loss_in_bins_test[continuous_array[k]
                                           ] += vectorial_loss[k]

    def update_ema_weights(self):
        for ema_param, param in zip(self.ema_weights, self.model.parameters()):
            if param.requires_grad:
                ema_param -= (1 - self.ema_rate) * (ema_param - param.detach())



