from kaldi.feat.fbank import Fbank, FbankOptions
from kaldi.feat.mfcc import Mfcc, MfccOptions
from kaldi.feat.wave import WaveData
from kaldi.matrix import Matrix, Vector, SubVector, SubMatrix
from kaldi.util.table import MatrixWriter
from espnet.asr.pytorch_backend.asr_init import load_trained_model
from kaldi.transform.cmvn import Cmvn

from speechbrain.pretrained import SpeakerRecognition

import scipy.io.wavfile
import matplotlib.pyplot as plt
import numpy as np
import torchaudio
import torch
from kaldi.feat.pitch import compute_and_process_kaldi_pitch, PitchExtractionOptions, ProcessPitchOptions

class PPGExtractor:
    def __init__(self, model_path):
        self.model, self.train_args = load_trained_model(model_path)
        self.model.eval()

    def extract(self, wav_file):
        # Load the waveform and extract filterbank features with pitch
        filterbank_pitch_features = extract_fbank(wav_file)
        
        # Extract the PPG features
        with torch.no_grad():
            ppg_features = self.model.encode(filterbank_pitch_features)
        
        return ppg_features

class XvectorExtractor:
    def __init__(self):
        self.model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmpdir_ecapa")
        self.model.eval()
    
    def extract(self, wav_file):
        signal, sample_rate = torchaudio.load(wav_file)
        xvector_embedding = self.model.encode_batch(signal).detach().cpu().numpy()
        return xvector_embedding


def compute_pitch_features(wav_data):
    pitch_opts = PitchExtractionOptions()
    pitch_opts.samp_freq = 16000
    post_opts = ProcessPitchOptions()
    wav_vector = Vector(wav_data)   
    pitch_features = compute_and_process_kaldi_pitch(pitch_opts, post_opts, wav_vector)
    return pitch_features.numpy()

def extract_fbank(wav_file, apply_cmvn=True):
    # Define Fbank options
    fbank_opts = FbankOptions() 
    fbank_opts.mel_opts.num_bins = 80           # Number of Mel filterbank bins
    fbank_opts.frame_opts.dither = 0
    
    fbank_extractor = Fbank(fbank_opts)
    sample_rate, waveform = scipy.io.wavfile.read(wav_file)
    waveform_vector = Vector(waveform)
    
    fbank_features = fbank_extractor.compute_features(waveform_vector, sample_rate, 1)
    pitch_features = compute_pitch_features(waveform)
    
    # Pad the shorter feature set to match the length of the longer one
    if pitch_features.shape[0] > fbank_features.shape[0]:
        pad_length = pitch_features.shape[0] - fbank_features.shape[0]
        fbank_features = np.pad(fbank_features, ((0, pad_length), (0, 0)), mode='constant')
    elif pitch_features.shape[0] < fbank_features.shape[0]:
        pad_length = fbank_features.shape[0] - pitch_features.shape[0]
        pitch_features = np.pad(pitch_features, ((0, pad_length), (0, 0)), mode='constant')
    
    combined_features = np.concatenate((fbank_features, pitch_features), axis=1)
    
    if apply_cmvn:
        cmvn_object = Cmvn()
        cmvn_object.read_stats("models/cmvn.ark")
        combined_features = Matrix(combined_features)
        cmvn_object.apply(combined_features, norm_vars=True)
        combined_features = combined_features.numpy()
    
    return combined_features
