import torch
import torch.nn as nn
from src.data import Collator

class ForwardReader:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate(self, input_ids, attention_mask, num_answers=5):
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=num_answers,
                num_return_sequences=num_answers
            )
        return outputs 
 
class BackwardReader:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate(self, input_ids, attention_mask, opt_info=None):
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=50,
                opt_info=opt_info
            )
        return outputs 

class CoherenceCollator(Collator):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        super(CoherenceCollator, self).__init__(text_maxlength, tokenizer, answer_maxlength=answer_maxlength)
    
    def __call__(self, batch):
        outputs = super().__call__(batch)
        coherence_outputs = outputs + (batch,)
        return coherence_outputs

class CoherenceModel(nn.Module):
    def __init__(self, f_reader, b_reader):
        super(CoherenceModel, self).__init__()

        self.f_reader = f_reader
        self.b_reader = b_reader
    
        D = 768
        input_d = D * 2
        self.sub_match_f = nn.Sequential(
                        nn.Linear(input_d, D),
                        nn.ReLU(),
                        nn.Dropout(),
                        nn.Linear(D, 1)
                        )

    def forward(self, input_ids, attention_mask, labels):
        f_outputs = self.f_reader.generate(input_ids, )
          
        return

