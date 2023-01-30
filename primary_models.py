# A wrapper class for primary models
# Define custom primary models here

from utils import BB_MODEL_ID
from bb_model import BigBirdForNaturalQuestions
from transformers import BigBirdTokenizer
from prepare_data import prepare_inputs_hp

class Primary_Model():
    def __init__(self, model_path=None):
        self.model = None
        self.tk = None
    def __call__(self, batch):
        self.forward(batch)
    def forward(self, batch):
        self.model(batch)
    def prepare_data(self, dataset):
        pass

class BigBird_PM(Primary_Model):
    def __init__(self, model_path=None):
        self.tk = BigBirdTokenizer.from_pretrained(BB_MODEL_ID)
        if model_path is None:
            self.model = BigBirdForNaturalQuestions.from_pretrained(BB_MODEL_ID, self.tk)
        else:
            self.model = BigBirdForNaturalQuestions.from_pretrained(model_path, self.tk)
    def __call__(self, example):
        return self.model(example)
    def forward(self, example):
        return self.model(example)
    def prepare_data(self, dataset):
        return dataset.map(lambda x: prepare_inputs_hp(x, tk=self.tk, max_length=self.model.bert.embeddings.position_embeddings.weight.shape[0]))
