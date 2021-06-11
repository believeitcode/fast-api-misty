from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

from qna import *

import json


class QuestionRequest(BaseModel):
    question: str


class QuestionReponse(BaseModel):
    ans_start_index: int
    ans_end_index: int 
    start_token_score: float
    end_token_score: float
    answer: str 

# Using virtualenv: virtualenv <env> , To activate virutalenv: <env>\Scripts\activate
# start server to run instant of FastAPI by 'uvicorn api:app --reload' (api - script file ,app - var instant of FastAPI)  
# ref: https://www.tutlinks.com/create-and-deploy-fastapi-app-to-heroku/ 
app = FastAPI()

with open("config.json") as json_file:
    config = json.load(json_file)

#root go to http://127.0.0.1:8000/docs for API Doc Swagger UI
@app.get('/')
def hello_world():
    return{"Hello": "World"}

#Tested with POSTMAN HTTP POST with body {"question":"string"}
@app.post('/qna/', response_model=QuestionReponse, response_model_include={"answer"}) # response_model_exclude -> to exclude certain fields
def predict(request: QuestionRequest):
    print(request.question)
    ans_start_id, ans_end_id , start_token_s , end_token_s, ans = bert_answering_machine(request.question, config["CONTEXT"])
    return QuestionReponse(
        ans_start_index= ans_start_id,
        ans_end_index=ans_end_id,
        start_token_score=start_token_s,
        end_token_score=end_token_s,
        answer= ans
    )