
import numpy as np 
import gtts
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pydub import AudioSegment
from scipy.ndimage.filters import maximum_filter, minimum_filter
import seaborn as sns
import os
import boto3



          


import crepe
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

def polly(word, fname):

    aws_access_key_id = os.environ.get('POLLY_API_ID')
    aws_secret_access_key = os.environ.get('POLLY_API_KEY')  
    
    polly_client = boto3.Session(
                    aws_access_key_id=aws_access_key_id,                     
        aws_secret_access_key=aws_secret_access_key ,
        region_name='us-west-2').client('polly')

    response = polly_client.synthesize_speech(VoiceId='Vicki',
                    OutputFormat='mp3', 
                    Text = word,
                    Engine = 'neural')

    file = open(fname+'.mp3', 'wb')
    file.write(response['AudioStream'].read())
    file.close()
   



class wordspectrum:
    """ Convert a word into an object of features and display them
    Keyword arguments: 
    word -- word that should be computed
    features -- which features to show, either 'mel' or 'mfccs'
    order -- order of the derivative of the feature
    thresh -- minimum intensity of the signal at the beginning and end. Bevore or after the waveform will be cut
    normalize -- if the features should get normalized. True or False, default is True
    n_mfcc -- number of n_mfcc features to calculate
    lang -- language of the word
    int_res = length of the window used for the intensity calculation in seconds
    """
    
    def __init__(self,word,path = 'audio', 
    api = 'aws',
    redo_audio = False,
    calc_mfccs = True,
    calc_mel = False,
    calc_intensity = False,
    calc_pitch = False,
    del_audio = False, 
    features = 'mel', 
    max_order = 2, 
    thresh=0.01,
    normalize=True, 
    n_mfcc = 20, 
    lang='de',
    int_res=0.01):
        self.path_local = os.path.dirname(__file__)
        path = os.path.join(self.path_local, path)
        self.word = word.lower()
        self.fname = os.path.join(path, word)
        self.api = api
        self.redo_audio = redo_audio

        self.calc_mel = calc_mel
        self.calc_mfccs = calc_mfccs
        self.calc_intensity = calc_intensity
        self.calc_pitch = calc_pitch

        self.max_order = max_order
        self.y, self.sr = self.word_to_spec(lang,thresh) 
        self.samples = self.y.shape[0]
        self.duration = self.samples/self.sr

        if self.calc_mfccs:
            self.mfccs = self.word_to_spectrogram(normalize, 'mfccs', n_mfcc)     
        if self.calc_mel:                   
            self.mel = self.word_to_spectrogram(normalize, 'mel')
        if self.calc_intensity:
            self.intensity = self.get_intensity(int_res)
            self.syllabs = self.syllabication()
        if del_audio:
            os.remove(self.fname + '.wav')

        if self.calc_pitch:
            self.pitch, self.pitch_conf = self.get_pitch()

        
    def word_to_spec(self, lang, thresh):                       # get the sample values and sample rate from a word via gtts

        if not os.path.isfile(self.fname+'.wav') or self.redo_audio: 
            
            if self.api == 'gtts':
                tts = gtts.gTTS(self.word,lang=lang)
                tts.save(self.fname+'.mp3')
            elif self.api == 'aws':
                polly(self.word,self.fname)
            sound = AudioSegment.from_mp3(self.fname+'.mp3')
            os.remove(self.fname + '.mp3')
            sound.export(self.fname+'.wav', format='wav')
            
        wavfn= self.fname+'.wav'
        y, sr = librosa.load(wavfn)

        non_zero = np.where(np.absolute(y)>thresh)
        y = y[np.amin(non_zero):np.amax(non_zero)]

        return y, sr
    
    def get_pitch(self):
        time,freq,conf,pitch = crepe.predict(self.y, self.sr, viterbi=True)
        
        print(conf.shape)
        return freq, conf
    
    def syllabication(self, min_db = 76, min_max_db = 85, min_len = 10):
        
        below_thresh = self.intensity < min_db
        
        struct = np.zeros(min_len) < 1
        below_thresh = binary_dilation(below_thresh,structure=struct)
        below_thresh = binary_erosion(below_thresh,structure=struct).astype(int)
        
        splits = np.where(np.absolute(np.ediff1d(below_thresh)) > 0.7)      
        splits = np.insert(splits[0],0,0)
        
        syllabs = []
        for i in range(1,splits.shape[0]-1,2):
            if (np.amax(self.intensity[splits[i]:splits[i+1]]) > min_max_db or
                      np.amax(self.intensity[splits[i-1]:splits[i]]) > min_max_db):
                      syllabs.append(((splits[i]+splits[i+1])/2)/self.sr)
        
        return np.asarray(syllabs)
        
    def get_intensity(self, int_res, ref_power=10**(-12)):                              # calculate the intensity graph

        len_window = round(int_res * self.sr)
        win = np.ones(len_window) / len_window
        power_db = 10 * np.log10(np.convolve(self.y**2, win, mode='same') / ref_power)
        return power_db
    
    
    def word_to_spectrogram(self,normalize, features, n_mfcc = 20):               # get feature-matrix and derivatives from waveform  
        spectrogram_orders = []
        if features == 'mfccs':
            spectrogram = librosa.feature.mfcc(y=self.y, sr=self.sr,n_mfcc=n_mfcc)
        elif features == 'mel':
            spectrogram = librosa.feature.melspectrogram(y=self.y, sr=self.sr)
            spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
            
        if normalize == True: 
            spectrogram -= np.mean(spectrogram,axis=0)
        spectrogram_orders.append(spectrogram )
        
        for i in range (self.max_order):
            spectrogram = librosa.feature.delta(spectrogram, order = i+1)
            spectrogram_orders.append(spectrogram)
        return spectrogram_orders
    
    
    def show_spectrogram(self, features = 'mel', order=0):                    # plot the spectrogram
        if features == 'mel':
            text = 'Mel Spectrogram'
            spectrogram = self.mel[order]
        elif features == 'mfccs':
            text = 'MFCC features'
            spectrogram = self.mfccs[order]
            
        fig, ax = plt.subplots()
        img = librosa.display.specshow(spectrogram, x_axis='time', ax=ax)
        fig.colorbar(img, ax=ax)
        ax.set(title= text)
    
    
    def show_waveform(self):                                              # plot the waveform and intensity
        y = self.y
        sr = self.sr
        max_height = np.amax(y)
        X = np.linspace(0,self.duration,y.shape[0])
        
        fig, ax = plt.subplots()

        # Plot linear sequence, and set tick labels to the same color
        ax.plot(X, y, color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        if self.calc_intensity:
            for i in range(self.syllabs.shape[0]):
                plt.vlines(self.syllabs[i], 0, max_height, colors='g')
        plt.ylabel("amplitude")
        plt.xlabel("time")

        if self.calc_intensity:
            ax2 = ax.twinx()

            ax2.plot(X, self.intensity, color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            plt.ylabel("intensity")
        
        ax.set(title= 'Waveform')
        plt.rcParams["figure.figsize"] = (15,5)
        plt.show()
        
    
    def show_pitch(self):
        y = self.pitch 
        sr = 0.01
        max_height = np.amax(y)
        X = np.linspace(0,self.duration,y.shape[0])
        
        fig, ax = plt.subplots()

        # Plot linear sequence, and set tick labels to the same color
        ax.plot(X, y, color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        for i in range(self.syllabs.shape[0]):
            plt.vlines(self.syllabs[i], 0, max_height, colors='g')
        plt.ylabel("pitch frequency")
        plt.xlabel("time")
        ax2 = ax.twinx()

        ax2.plot(X, self.pitch_conf, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        plt.ylabel("pitch confidence")
        
        ax.set(title= 'Pitch Frequency')
        plt.rcParams["figure.figsize"] = (15,5)
        plt.show()
        
    def show(self, features = 'mel'):                                   #show features + derivatives + waveform+ intensity
        spectrogram = []
        if features == 'mel' and self.calc_mel:
            text = 'Mel Spectrogram'
            spectrogram = self.mel

        elif features == 'mfccs' and self.calc_mfccs:
            text = 'MFCC features'
            spectrogram = self.mfccs
            
        y = self.y
        sr = self.sr

       
        if spectrogram:

            for i in range(len(spectrogram)):
                fig, ax = plt.subplots()
                img = librosa.display.specshow(spectrogram[i], x_axis='time', ax=ax)
                fig.colorbar(img, ax=ax)
                ax.set(title=text+' '+str(i)+'. order')

           

        self.show_waveform()

        if self.calc_pitch:
            self.show_pitch()

