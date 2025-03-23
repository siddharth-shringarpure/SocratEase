import os
import torch
import pickle

model_path = os.path.join(os.path.dirname(__file__), 'model', 'logical_model.pk')

with open (model_path, 'rb') as f:
    logical_model = pickle.load(f)


text_logical = " The logic is the foundation of rational thinking, ensuring consistency and eliminating contradiction. The objective reasoning guarantees truth if premises are correct while inductive reasoning forms patterns from observation. The objective reasoning suggests the most plausible explanation of limited evidence. Logic is key in philosophy, AI, and ethics, shaping decisions and preventing misinformation, mastering it helps in making informed choices daily."

pred_logical = logical_model.predict(text_logical)
print(pred_logical)