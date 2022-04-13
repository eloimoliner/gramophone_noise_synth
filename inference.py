import numpy as np
import torch
from scipy.integrate import solve_ivp


class GramophoneSampler:

    def __init__(self, model, sde):
        self.model = model
        self.sde = sde
        self.audio_len=model.args.audio_len

    def create_schedules(self, nb_steps, split_t, truncation_t=None):
        t_schedule = torch.arange(0, nb_steps + 1) / nb_steps
        t_schedule = (self.sde.t_max - self.sde.t_min) * \
            t_schedule + self.sde.t_min
        split= (self.sde.t_max - self.sde.t_min) * \
            split_t + self.sde.t_min
          

        split=int(split*nb_steps)
 
        sigma_schedule = self.sde.sigma(t_schedule)
        m_schedule = self.sde.mean(t_schedule)
        if truncation_t!=None:
            trunc= (self.sde.t_max - self.sde.t_min) * \
                truncation_t + self.sde.t_min
            trunc=int(trunc*nb_steps)
            return sigma_schedule, m_schedule, split, trunc
        else:
            return sigma_schedule, m_schedule, split

    def predict_unconditional(
        self,
        nb_steps,
        num_periods,
        taup=None,
        periods_separated=False
    ):
        if num_periods >1:
            assert taup !=None, "please specify taup"
        else:
            taup= 0

        with torch.no_grad():
            
            sigma, m, period_split = self.create_schedules(nb_steps,taup)
            print("period split at step ",period_split)
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #sample from prior
            audio = torch.randn((1,self.audio_len)).to(device)
            print(audio.device)
            print("Generating period 1")
            #start sampling from trunc
            for n in range( nb_steps- 1, 0, -1):
                # begins at t = 1 (n = nb_steps - 1)
                # stops at t = 2/nb_steps (n=1)
                #print(1,n)

                audio = m[n-1] / m[n] * audio + (m[n] / m[n-1] * (sigma[n-1])**2 / sigma[n] - m[n-1] / m[n] * sigma[n]) * \
                    self.model(audio, sigma[n])

                if n > 0:  # everytime
                    noise = torch.randn_like(audio)
                    audio += sigma[n-1]*(1 - (sigma[n-1]*m[n] /
                                              (sigma[n]*m[n-1]))**2)**0.5 * noise

                if n==period_split:#divide time
                    audio_period=torch.clone(audio)

            # The noise level is now sigma(1/nb_steps) = sigma[0]
            # Jump step
            audio = (audio - sigma[0] * self.model(audio,
                                                   sigma[0])) / m[0]
            
            periodlist=[audio]

            for j in range(num_periods-1):
                print("Generating period ", str(j+2))
                audio=audio_period
                for n in range(period_split-1, 0, -1): #sample from the split stage
                    # begins at t = split_step  (n = period_split - 1)
                    # stops at t = 2/nb_steps (n=1)
                    #print(j+1,n)

                    audio = m[n-1] / m[n] * audio + (m[n] / m[n-1] * (sigma[n-1])**2 / sigma[n] - m[n-1] / m[n] * sigma[n]) * \
                        self.model(audio, sigma[n])

                    if n > 0:  # everytime
                        noise = torch.randn_like(audio)
                        audio += sigma[n-1]*(1 - (sigma[n-1]*m[n] /
                                                  (sigma[n]*m[n-1]))**2)**0.5 * noise
                audio = (audio - sigma[0] * self.model(audio,
                                                sigma[0])) / m[0]
                periodlist.append(audio)

        if not(periods_separated):
            res=torch.cat([periodlist[i] for i in range(0,num_periods)],dim=1)
        else:
            res=torch.cat([periodlist[i] for i in range(0,num_periods)],dim=0)

        return res

    def predict_conditional(
        self,
        audio,
        nb_steps,
        num_periods,
        taup,
        tau0,
        periods_separated=False
    ):

        with torch.no_grad():

            sigma, m, period_split, trunc = self.create_schedules(nb_steps,taup, tau0)
            print("truncation at step ", trunc)
            print("period split at step ",period_split)
            
            #map audio to latent space 
            n=trunc
            audio = m[trunc-1] * audio + sigma[trunc-1] * torch.randn_like(audio)
            if period_split>=trunc:
                period_split=trunc-1
            print("Generating period 1")
            #start sampling from trunc
            for n in range(trunc - 1, 0, -1):
                # begins at t = 1 (n = nb_steps - 1)
                # stops at t = 2/nb_steps (n=1)
                #print(1,n)

                audio = m[n-1] / m[n] * audio + (m[n] / m[n-1] * (sigma[n-1])**2 / sigma[n] - m[n-1] / m[n] * sigma[n]) * \
                    self.model(audio, sigma[n])

                if n > 0:  # everytime
                    noise = torch.randn_like(audio)
                    audio += sigma[n-1]*(1 - (sigma[n-1]*m[n] /
                                              (sigma[n]*m[n-1]))**2)**0.5 * noise

                if n==period_split:#divide time
                    audio_period=torch.clone(audio)
            # The noise level is now sigma(1/nb_steps) = sigma[0]
            # Jump step
            audio = (audio - sigma[0] * self.model(audio,
                                                   sigma[0])) / m[0]
            
            periodlist=[audio]

            for j in range(num_periods-1):
                print("Generating period ", str(j+2))
                audio=audio_period
                for n in range(period_split-1, 0, -1): #sample from the split stage
                    # begins at t = split_step  (n = period_split - 1)
                    # stops at t = 2/nb_steps (n=1)
                    #print(j+1,n)

                    audio = m[n-1] / m[n] * audio + (m[n] / m[n-1] * (sigma[n-1])**2 / sigma[n] - m[n-1] / m[n] * sigma[n]) * \
                        self.model(audio, sigma[n])

                    if n > 0:  # everytime
                        noise = torch.randn_like(audio)
                        audio += sigma[n-1]*(1 - (sigma[n-1]*m[n] /
                                                  (sigma[n]*m[n-1]))**2)**0.5 * noise
                audio = (audio - sigma[0] * self.model(audio,
                                                sigma[0])) / m[0]
                periodlist.append(audio)
        if not(periods_separated):
            res=torch.cat([periodlist[i] for i in range(0,num_periods)],dim=1)
        else:
            res=torch.cat([periodlist[i] for i in range(0,num_periods)],dim=0)
        return res


class Sampling:
    """
    DDPM-like discretization of the SDE as in https://arxiv.org/abs/2107.00630
    This is the most precise discretization
    """

    def __init__(self, model, sde):
        self.model = model
        self.sde = sde

    def create_schedules(self, nb_steps):
        t_schedule = torch.arange(0, nb_steps + 1) / nb_steps
        t_schedule = (self.sde.t_max - self.sde.t_min) * \
            t_schedule + self.sde.t_min
        sigma_schedule = self.sde.sigma(t_schedule)
        m_schedule = self.sde.mean(t_schedule)

        return sigma_schedule, m_schedule

    def predict(
        self,
        audio,
        nb_steps
    ):

        with torch.no_grad():

            sigma, m = self.create_schedules(nb_steps)

            for n in range(nb_steps - 1, 0, -1):
                # begins at t = 1 (n = nb_steps - 1)
                # stops at t = 2/nb_steps (n=1)

                audio = m[n-1] / m[n] * audio + (m[n] / m[n-1] * (sigma[n-1])**2 / sigma[n] - m[n-1] / m[n] * sigma[n]) * \
                    self.model(audio, sigma[n])

                if n > 0:  # everytime
                    noise = torch.randn_like(audio)
                    audio += sigma[n-1]*(1 - (sigma[n-1]*m[n] /
                                              (sigma[n]*m[n-1]))**2)**0.5 * noise

            # The noise level is now sigma(1/nb_steps) = sigma[0]
            # Jump step
            audio = (audio - sigma[0] * self.model(audio,
                                                   sigma[0])) / m[0]

        return audio


