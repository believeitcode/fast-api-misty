


from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import torch
import numpy as np
import json 

## Commented Area are downloading model from Hugging Face only applied on first use only
## check out for more info : https://huggingface.co/mrm8488/bert-tiny-finetuned-squadv2 
with open("config.json") as json_file:
   config = json.load(json_file)

PATH = "bert-tiny-finetuned-squadv2" 

# model = BertForQuestionAnswering.from_pretrained(config["BERT_MODEL"])
# tokenizer_for_bert = BertTokenizer.from_pretrained(config["BERT_MODEL"])

# tokenizer_for_bert.save_pretrained(PATH)
# model.save_pretrained(PATH)


model = BertForQuestionAnswering.from_pretrained(PATH)
tokenizer_for_bert = BertTokenizer.from_pretrained(PATH)

def bert_answering_machine ( question, passage, max_len =  512):
    ''' Function to provide answer from passage for question asked.
        This function takes question as well as the passage 
        It retuns answer from the passage, along with start/end token index for the answer and start/end token scores
        The scores can be used to rank answers if we are searching answers for same question in multiple passages
        Value of max_len can not exceed 512. If length of question + passage + special tokens is bigger than max_len, function will truncate extra portion.
        
    '''
  
    #Tokenize input question and passage. Keeping maximum number of tokens as specified by max_len parameter. This will also add special tokens - [CLS] and [SEP]
    input_ids = tokenizer_for_bert.encode ( question, passage,  max_length= max_len, truncation=True)  
    
    
    #Getting number of tokens in 1st sentence (question) and 2nd sentence (passage)
    cls_index = input_ids.index(102) #Getting index of first SEP token
    len_question = cls_index + 1       # length of question (1st sentence)
    len_answer = len(input_ids)- len_question  # length of answer (2nd sentence)
    
    
    #BERT need Segment Ids to understand which tokens belong to sentence 1 and which to sentence 2
    segment_ids =  [0]*len_question + [1]*(len_answer)  #Segment ids will be 0 for question and 1 for answer
    
    #Converting token ids to tokens
    tokens = tokenizer_for_bert.convert_ids_to_tokens(input_ids) 
    
    
    # getting start and end scores for answer. Converting input arrays to torch tensors before passing to the model
    start_token_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]) )[0]
    end_token_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]) )[1]

    #Converting scores tensors to numpy arrays so that we can use numpy functions
    start_token_scores = start_token_scores.detach().numpy().flatten()
    end_token_scores = end_token_scores.detach().numpy().flatten()
    
    #Picking start index and end index of answer based on start/end indices with highest scores
    answer_start_index = np.argmax(start_token_scores)
    answer_end_index = np.argmax(end_token_scores)

    #Getting scores for start token and end token of the answer. Also rounding it to 2 decimal digits
    start_token_score = np.round(start_token_scores[answer_start_index], 2)
    end_token_score = np.round(end_token_scores[answer_end_index], 2)
    
   
    #Combining subwords starting with ## so that we can see full words in output. Note tokenizer breaks words which are not in its vocab.
    answer = tokens[answer_start_index] #Answer starts with start index, we got based on highest score
    for i in range(answer_start_index + 1, answer_end_index + 1):
        if tokens[i][0:2] == '##':  # Token for a splitted word starts with ##
            answer += tokens[i][2:] # If token start with ## we remove ## and combine it with previous word so as to restore the unsplitted word
        else:
            answer += ' ' + tokens[i]  # If token does not start with ## we just put a space in between while combining tokens
            
    # Few patterns indicating that BERT does not get answer from the passage for question asked
    if ( answer_start_index == 0) or (start_token_score < 0 ) or  (answer == '[SEP]') or ( answer_end_index <  answer_start_index):
        answer = "Sorry!, I could not find  an answer in the passage."
    
    return ( answer_start_index, answer_end_index, start_token_score, end_token_score,  answer)