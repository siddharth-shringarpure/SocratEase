import pickle
import numpy as np
import os

model_path = os.path.join(os.path.dirname(__file__), 'model', 'tonality_model.pk')
with open(model_path, 'rb') as f:
    model = pickle.load(f)
model_path = os.path.join(os.path.dirname(__file__), 'model', 'tonality_embedding.pk')
with open(model_path, 'rb') as f:
    modelEmbedding = pickle.load(f)
model_path = os.path.join(os.path.dirname(__file__), 'model', 'tonality_LE.pk')
with open(model_path, 'rb') as f:
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