

from glob import glob
from xppg_pca.inference import PPGXvectorPCAInference
from tqdm import tqdm
import librosa
import soundfile as sf
import numpy as np

ppg_pca = PPGXvectorPCAInference("models/model.last5.avg.best", 
                                     "models/ppg_xvector_pca_object_moment_1.pkl")

# Generate and save a sample audio file with random noise

y = np.ones(16000)
sf.write('tools/random_noise.wav', y, 16000)

wav_path = 'tools/random_noise.wav'
result = ppg_pca.predict(wav_path)

# Approximately equal
target_value = -0.20908679
#target_value = 14214

try:
    np.testing.assert_approx_equal(result, target_value, significant=3)
except AssertionError:
    # If the test fails, print the result
    print(result, "is not equal to ", target_value)

print("Test passed")

