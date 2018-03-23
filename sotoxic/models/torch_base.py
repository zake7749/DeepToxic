import numpy as np
import torch
import torch.nn as nn
import importlib

from sotoxic.utils import generators
importlib.reload(generators)  # for debugging on jupyter

from keras.models import Model


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()
        print("Choose the torch base model.")
        self.manager = ModelManager()

    def save(self, path):
        self.manager.save_model(self, path)

    def load(self, path):
        self.manager.load_model(self, path)

    def forward(self, x):
        raise NotImplementedError

    def predict(self, x, batch_size=256, verbose=0):
        self.eval_mode()
        predictions = []
        for batch_x in generators.test_batches_generator(x, batch_size):
            preds_var = self.forward(batch_x)
            preds_logits = nn.Sigmoid()(preds_var)
            predictions.append(preds_logits.data.cpu().numpy())
        predictions = np.concatenate(predictions, axis=0)
        return predictions

    def set_dropout(self, p):
        pass

    def train_mode(self, p):
        self.set_dropout(p)
        self.train()

    def eval_mode(self):
        self.eval()


class ModelManager(object):

    def __init__(self, path=None):
        self.path = path

    def save_model(self, model, path=None):
        path = self.path if path is None else path
        torch.save(model.state_dict(), path)
        print("Model has been saved as %s.\n" % path)

    def load_model(self, model, path=None):
        path = self.path if path is None else path
        model.load_state_dict(torch.load(path))
        model.eval()
        print("A pre-trained model at %s has been loaded.\n" % path)