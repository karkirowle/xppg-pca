import os
import warnings
import pickle as pkl
import numpy as np
import kaldiio
from collections import defaultdict
from scipy.stats import pearsonr

def ppg_xvector_pca(file):
    with open(f'models/ppg_xvector_pca_object_moment_1.pkl', 'rb') as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pca_object = pkl.load(f)
                    
    key = os.path.splitext(os.path.basename(file))[0]
    scp_file = "static_features/ppg.scp"
    d = kaldiio.load_scp(scp_file)
    
    ppg = d[key]
    xvec_file = "static_features/COPAS_dict.npy"
    xvec_dict = np.load(xvec_file, allow_pickle=True).item()
    xvec = xvec_dict[key]
    
    ppg_norm = np.mean(ppg, axis=0)
    ppg_norm = ppg_norm[None, :] / np.linalg.norm(ppg_norm)
    xvec_norm = xvec[None, :] / np.linalg.norm(xvec)
    feature = np.concatenate((ppg_norm, xvec_norm), axis=1)
    pca_component = pca_object.transform(feature)[0,0]
    
    
    return pca_component


def process_copas(speaker_category = "D"):
    
    xvec_file = "static_features/COPAS_dict.npy"
    xvec_dict = np.load(xvec_file, allow_pickle=True).item()
    
    keys = xvec_dict.keys()
    # Select only S1-S2 keys
    keys = [key for key in keys if "S1" in key or "S2" in key]
    
    speakers = set([os.path.basename(key).split("_")[0] for key in keys])
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
        
        speaker_keys = [key for key in keys if speaker == os.path.basename(key).split("_")[0]]
        
        results = [ppg_xvector_pca(key) for key in speaker_keys]
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

def report_results():
    
    speaker_categories = ["V", "L", "H", "D", "all", "all_except_D"]
    severity_file = "static_features/speaker_scores"
    severity_dict = load_severity_scores_copas(severity_file)

    for category in speaker_categories:
        results = process_copas(category)
        r, p = calculate_correlations(severity_dict, results)
        
        print("Category: ", category, "Correlation: ", r, "p-value: ", p)
    


if __name__ == "__main__":

    speaker_category = "D"  # or any other category you want to process
    report_results()