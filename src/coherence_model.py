import torch
import torch.nn as nn
from src.data import Collator
import src.evaluation
import copy
from src.data import get_backward_question

class ForwardReader:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()

    def generate(self, input_ids, attention_mask, batch_examples, num_answers=5):
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=num_answers,
                num_return_sequences=num_answers
            )
        batch_size = len(input_ids)
        assert(len(outputs) == (batch_size * num_answers))
        outputs = outputs.reshape(batch_size, num_answers, -1)
        for b_idx, answer_code_lst in enumerate(outputs):
            example = batch_examples[b_idx]
            top_pred_info_lst = []
            for answer_idx, answer_code in enumerate(answer_code_lst):
                pred_answer = self.tokenizer.decode(answer_code, skip_special_tokens=True)
                em_score = None
                if 'answers' in example:
                    em_score = src.evaluation.ems(pred_answer, example['answers'])
                
                f_pred_info = {
                    'answer':pred_answer,
                    'em':em_score
                }
                top_pred_info_lst.append(f_pred_info)
                 
            example['top_preds'] = top_pred_info_lst

 
class BackwardReader:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()

    def generate(self, input_ids, attention_mask, batch_examples, opt_info=None):
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=50,
                opt_info=opt_info
            )
         
        for b_idx, answer_code in enumerate(outputs):
            pred_answer = self.tokenizer.decode(answer_code, skip_special_tokens=True)
            b_example = batch_examples[b_idx]
               

        return outputs 

class CoherenceCollator(Collator):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        super(CoherenceCollator, self).__init__(text_maxlength, tokenizer, answer_maxlength=answer_maxlength)
    
    def __call__(self, batch):
        outputs = super().__call__(batch)
        coherence_outputs = outputs + (batch,)
        return coherence_outputs

class CoherenceModel(nn.Module):
    def __init__(self, f_reader, b_reader, collator):
        super(CoherenceModel, self).__init__()
        self.f_reader = f_reader
        self.b_reader = b_reader
        self.collator = collator
    
        D = 768
        input_d = D * 2
        self.sub_match_f = nn.Sequential(
                        nn.Linear(input_d, D),
                        nn.ReLU(),
                        nn.Dropout(),
                        nn.Linear(D, 1)
                        )

    def forward(self, input_ids, attention_mask, labels, batch_examples):
        if not 'top_preds' in batch_examples[0]: 
            self.f_reader.generate(input_ids, attention_mask, batch_examples)
        
        backward_example_lst = [] 
        for example in batch_examples:
            forward_preds = example['top_preds']
            b_example_lst = []
            for f_pred in forward_preds:
                if 'back_example' not in f_pred:
                    b_example = copy.deepcopy(example)
                    del b_example['top_preds'] 
                    self.construct_back_info(b_example)
                    f_pred['back_example'] = b_example

                b_example_lst.append(f_pred['back_example'])
            
            b_batch = self.collator(b_example_lst)
            (_, _, _, context_ids, context_mask, _) = b_batch
            b_input_ids=context_ids.cuda()
            b_attention_mask=context_mask.cuda()
            self.b_reader.generate(b_input_ids, b_attention_mask, b_example_lst, opt_info={})
            b_sub_state_lst = opt_info['answer_state'] 
          
        return

    def construct_back_info(self, b_example):
        question = b_example['example_question']
        subject = b_example['example_subject']
        target = b_example['example_target']
        back_question = get_backward_question(question, subject, target)
        b_example['subject'] = target
        
        b_example['target'] = subject + ' </s>'
        b_example['question'] = b_example['question_prefix'] + " " + back_question
        
        del b_example['subject']
        del b_example['example_question']
        del b_example['example_subject']
        del b_example['example_target']
        del b_example['question_prefix']
         
