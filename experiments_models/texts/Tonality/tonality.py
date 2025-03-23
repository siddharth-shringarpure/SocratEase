import pickle
import numpy as np
with open('tonality_model.pk', 'rb') as f:
    model = pickle.load(f)
with open('tonality_embedding.pk', 'rb') as f:
    modelEmbedding = pickle.load(f)
with open('tonality_LE.pk', 'rb') as f:
    LE = pickle.load(f)
def getEmbeddings (text):
    res = modelEmbedding.encode(text)
    return res
def tonality(text):
    embedding = getEmbeddings(text)
    x = np.array(embedding).reshape(1, -1)
    pred = model.predict(x)
    pred = LE.inverse_transform(pred)
    return pred[0]
pred = tonality("You fucking piece of shit!")

print(pred)