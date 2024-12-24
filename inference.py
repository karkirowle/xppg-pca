import pickle as pkl
import os
import numpy as np
import warnings
from glob import glob
from collections import defaultdict
from utils import PPGExtractor, XvectorExtractor
from params import COPAS_PATH
from scipy.stats import pearsonr


class PPGXvectorPCAInference:
    def __init__(self, model_path, pca_model_path):
        self.ppg_extractor = PPGExtractor(model_path=model_path)
        self.xvector_extractor = XvectorExtractor()
        with open(pca_model_path, 'rb') as f:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.pca_object = pkl.load(f)

    def predict(self, file):
        ppg = self.ppg_extractor.extract(file).detach().numpy()
        xvec = self.xvector_extractor.extract(file)[0, 0, :]

        ppg_norm = np.mean(ppg, axis=0)
        ppg_norm = ppg_norm[None, :] / np.linalg.norm(ppg_norm)
        xvec_norm = xvec[None, :] / np.linalg.norm(xvec)
        feature = np.concatenate((ppg_norm, xvec_norm), axis=1)
        pca_component = self.pca_object.transform(feature)[0, 0]

        return pca_component



def process_copas(speaker_category, ppg_pca):
    
    files = glob(f"{COPAS_PATH}/Data/Data/S1/*.wav")
    files.extend(glob(f"{COPAS_PATH}/Data/Data/S2/*.wav"))

    speakers = set([os.path.basename(file).split("_")[0] for file in files])
    
    speaker_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for speaker in speakers:
        
        if speaker_category == "all":
            if "N" in speaker:
                continue
        elif speaker_category == "all_except_D":
            if ("D" in speaker) or ("N" in speaker):
                continue
        else:
            if speaker_category not  in speaker:
                continue
        
        files = glob(f"{COPAS_PATH}/Data/Data/S1/{speaker}_*.wav")
        files.extend(glob(f"{COPAS_PATH}/Data/Data/S2/{speaker}_*.wav"))

        results = [ppg_pca.predict(file) for file in files]
        speaker_results[f"{speaker}"] = np.mean(results)
    
   
    return speaker_results
    
    
def calculate_correlations(severity_dict, results_dict):
    severities, scores = [], []
    speaker_list = list()
    for key in results_dict:
        if key in severity_dict:
            severities.append(severity_dict[key])
            scores.append(results_dict[key])
            speaker_list.append(key.split("_")[0])

    return pearsonr(severities, scores)

def load_severity_scores_copas(severity_file):
    severity_dict = {}
    with open(severity_file, "r") as f:
    
        for line in f:
            entries = line.strip().split(",")
            rating = entries[4]
            speaker = entries[0]
            severity_dict[speaker] = float(rating)
    
    return severity_dict

def report_results(ppg_pca):
    
    speaker_categories = ["V", "L", "H", "D", "all", "all_except_D"]
    severity_file = "static_features/speaker_scores"
    severity_dict = load_severity_scores_copas(severity_file)

    for category in speaker_categories:
        results = process_copas(category, ppg_pca)
        r, p = calculate_correlations(severity_dict, results)
        
        print("Category: ", category, "Correlation: ", r, "p-value: ", p)
    
#main part

if __name__ == "__main__":
    
    
    import argparse
    parser = argparse.ArgumentParser()
    
    # Have a flag to run on the COPAS otherwise use run on an audio file
    parser.add_argument("--copas", action="store_true")
    parser.add_argument("--audio_file", type=str, default=f"{COPAS_PATH}/Data/Data/S1/S1_1.wav")
    args = parser.parse_args()
    
    
    ppg_pca = PPGXvectorPCAInference("models/model.last5.avg.best", 
                                     "models/ppg_xvector_pca_object_moment_1.pkl")
    

    
    if args.copas:
        report_results(ppg_pca)
    else:
        
        score = ppg_pca.predict(args.audio_file)
        print(score)

    