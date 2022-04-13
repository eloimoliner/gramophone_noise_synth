import math
import numpy as np
from scipy.fft import fft, ifft
class noise_presynthesis:
    def __init__(self, gains=[[0,0,0,0,0,0]],
                 noise_gain=0, args=None
                 ):
        #save params
        self.gains=gains
        self.noise_gain=noise_gain
        self.args=args
        self.Td=60/78
        self.generate_hiss_EQ()
        


    def generate_hiss_EQ(self):
        
        #Parametric equalizer with sheliving and peaking filters. Implementation based on torchaudio. May not be very efficient

        gains=np.array(self.gains)
        shape=gains.shape
        num_variations=shape[0]
        #freqs=[125,250,500,1000,2000,4000,8000,16000]
        freqs=[125,500,2000,8000,16000]
        assert len(freqs)==shape[1]
        #Q=0.98
        lens=[int(self.args.audio_len/num_variations) for i in range(num_variations)]
        lens[-1]= lens[-1]+ self.args.audio_len-sum(lens)

        NFFT = pow(2, math.ceil(math.log(self.args.audio_len)/math.log(2)));

        for v in range(0,num_variations):
            #xv = 10**((self.noise_gain)/10)* torch.randn(1, NFFT, device=device)
            xv = 10**((self.noise_gain)/10)*np.random.randn(NFFT)

            H=np.zeros((int(NFFT/2 +1)))
            fi=int(freqs[0]*NFFT/self.args.sample_rate)
            H[0:fi]=10**(gains[v][0]/10)
            for i in range(1,len(freqs)-1):
                fi2=int(freqs[i]*NFFT/self.args.sample_rate)
                H[fi:fi2]=np.linspace(10**(gains[v][i-1]/10),10**(gains[v][i]/10),fi2-fi)
                fi=fi2

            fi2=int(freqs[-1]*NFFT/self.args.sample_rate)
            H[fi:fi2]=np.linspace(10**(gains[v][-2]/10),10**(gains[v][-1]/10),fi2-fi)
            H[fi2::]=10**(gains[v][-1]/10)
            H=np.concatenate((H,np.flip(H[1:-1])))

            X=fft(xv)
            X=X*H
            xv=ifft(X)
            xv=np.real(xv)

            xv=xv[0:lens[v]]
                            
            if v==0:
                x=xv
            else:
                x=np.concatenate((x,xv))
        self.x=np.roll(x, int(self.args.audio_len/(2*num_variations))) 
        self.x=self.x.astype('float32')


    
    def normalize(self):
        b=1.4826
        self.sigma=np.sqrt(b**2 * np.median(self.x**2))
        self.x=(0.1/self.sigma)*self.x

    def unnormalize(self,noise):
        return (self.sigma/0.1)*noise

    def synthesize_thump(self,pos,strength=2):
        if strength==1:
            #level soft
            M=0.2 #in ms length of the discontinuity
            M=int(self.args.sample_rate*M*1e-3)
            Aclick=0.03*12
            Atail=0.025*10
            tau_e=0.02 #(seconds)
            tau_f= 0.015
            fmax=60
            fmin=20
        elif strength==2:
            #level soft
            M=0.4 #in ms length of the discontinuity
            M=int(self.args.sample_rate*M*1e-3)
            Aclick=0.03*17
            Atail=0.025*15
            tau_e=0.03 #(seconds)
            tau_f= 0.015
            fmax=60
            fmin=20
        elif strength==3:
            #level soft
            M=1 #in ms length of the discontinuity
            M=int(self.args.sample_rate*M*1e-3)
            Aclick=0.03*24
            Atail=0.025*25
            tau_e=0.06 #(seconds)
            tau_f= 0.015
            fmax=60
            fmin=20
        elif strength==4:
            #level normal
            M=2 #in ms length of the discontinuity
            M=int(self.args.sample_rate*M*1e-3)
            Aclick=0.03*35
            Atail=0.025*35
            tau_e=0.08 #(seconds)
            tau_f= 0.015
            fmax=60
            fmin=20
        elif strength==5:
            #level normal
            M=2 #in ms length of the discontinuity
            M=int(self.args.sample_rate*M*1e-3)
            Aclick=0.03*37
            Atail=0.025*39
            tau_e=0.10 #(seconds)
            tau_f= 0.015
            fmax=60
            fmin=20

        
        L=int(self.args.audio_len- M)
        n=np.linspace(0,L-1, L)
        fn=(fmax-fmin)*np.exp(-n/(self.args.sample_rate* tau_f)) +fmin
        tail=Atail*  np.exp(-n/(self.args.sample_rate*tau_e)) *np.sin(2*math.pi*n*fn/self.args.sample_rate   -math.pi/4)
        #Just modeling the tail, let's see if we will need to model the click as well

        thump=np.concatenate((Aclick*np.random.randn(M),tail))
        thump=np.roll(thump, pos)
        return thump

    def add_thumps(self,positions, strengths):
        for i in range(len(positions)):
            s=int(positions[i]*self.Td*self.args.sample_rate)
            thump=self.synthesize_thump(s,strengths[i])
            self.x+=thump
    

    def add_buzz(self,pow, sharpness=0, f=60):
        t=np.linspace(0,self.args.audio_len-1, self.args.audio_len)
        sin=np.sin(2*math.pi*t*f/self.args.sample_rate)

        if sharpness>0:
            d=sharpness/2+0.5
            c=-4*d+2
            b=1-c 
            sin=b*sin+c*sin**2
        
        rms=np.sqrt(np.var(sin))

        self.x+=(10**(pow/10)/rms)*sin
