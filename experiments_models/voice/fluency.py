import librosa
import pickle
import torch
import nltk
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import re

# Download NLTK tokenizer (if not already downloaded)
nltk.download('punkt_tab')

# Initialize stemmer
stemmer = PorterStemmer()

def preprocess_text(text):
    text = str(text).lower()  
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = word_tokenize(text)  # Tokenize into words
    stemmed_words = [stemmer.stem(word) for word in words]

    count_repeated = Counter(stemmed_words)
    more_than_once = {key: count for key, count in count_repeated.items() if count > 1}
    return len(more_than_once), len(stemmed_words)
    
def calculate_ttr(text):
    words = re.findall(r'\b\w+\b', text.lower())
    
    unique_words = set(words)

    ttr = len(unique_words) / len(words)
    return ttr
    
def mfcc(df):
    mfcc = librosa.feature.mfcc(y = df['audio_signal'], sr = sr , n_mfcc = n_mfcc)
    for i in range(0, n_mfcc):
        df[f'mfcc_{i+1}(mean)'] = mfcc[i].mean()
        df[f'mfcc_{i+1}(std)'] = mfcc[i].std()
    return df
    
def zeroCrossingRate(df):
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y = df['audio_signal'], hop_length = hop_length)[0]
    df['ZCR(mean)'] = zero_crossing_rate.mean()
    df['ZCR(std)'] = zero_crossing_rate.std()
    return df
    
def fundamental_frequency(df):
    f0, a, b = librosa.pyin(df['audio_signal'],sr=sr,fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0 = f0[~np.isnan(f0)]
    if len(f0) != 0:
        max_frequency = f0.max()
        min_frequency = f0.min()
        std_frequency = f0.std()
        median_frequency = np.median(f0)
    else:
        median_frequency = 0
        max_frequency = 0
        min_frequency = 0
        std_frequency = 0
    df['f0_max'] = max_frequency
    df['f0_min'] = min_frequency
    df['f0_median'] = median_frequency
    df['f0_std'] = std_frequency
    return df

whisper = torch.load('whisper_model.pt', weights_only = False)
with open ('voice_classifier.pk', 'rb') as f:
    classifier = pickle.load(f)
with open ('scaler.pk', 'rb') as f:
    scaler = pickle.load(f)
with open ('voice_labeler.pk', 'rb') as f:
    LE = pickle.load(f)

transcribe = whisper.transcribe('sample.wav')
signal, sr = librosa.load('sample.wav', sr = 16000)

duration_seconds = len(signal)/sr

text = transcribe['text']
one_gram_repeat, num_words_spoken = preprocess_text(text)

hop_length = 512
n_mfcc = 25


df = pd.DataFrame()
df['speech_rate'] = [(num_words_spoken/duration_seconds)*60]
df['lexical_density'] = [calculate_ttr(text)]
df['1gram_repeat'] = [one_gram_repeat]
df['audio_signal'] = [signal]
df = df.apply(zeroCrossingRate, axis=1)
df = df.apply(mfcc, axis=1)
df = df.apply(fundamental_frequency, axis=1)

df = df.drop(columns = ['audio_signal'])

X = pd.DataFrame(scaler.transform(df), columns=df.columns)
y = classifier.predict(X)
y = LE.inverse_transform(y)
print(y)
    
