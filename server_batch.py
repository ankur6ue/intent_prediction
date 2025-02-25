import requests
import json
from fastapi import FastAPI
from ray import serve
import time
import logging
import torch
import numpy as np
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from pydantic import BaseModel
from typing import List
logger = logging.getLogger("ray.serve")


class InputModel(BaseModel):
    text: str

class OutputModel(BaseModel):
    prediction: str
    scores: List[float]
    inference_time: float


sentence = "add this song to the album"
app = FastAPI()

@serve.deployment(
    num_replicas="auto",
    max_ongoing_requests=30,
    autoscaling_config={"min_replicas": 1,
                        "upscale_delay_s": 10,
                        "downscale_delay_s": 5,
                        "metrics_interval_s": 5,
                        "max_replicas": 6},
    ray_actor_options={"num_cpus": 2, "num_gpus": 0})

@serve.ingress(app)
class DetermineIntent:
    def __init__(self):
        # limit number of threads.. no need to do the below manually. It seems to be done either by pytorch or ray
        # based on the num_cpus in the serve.deployment configuration
        # num_threads = 5
        # torch.set_num_threads(num_threads)
        # torch.set_num_interop_threads(num_threads)
        # Load model
        logger.debug("loading the model..")
        self.label2id = json.load(open('/home/ankur/dev/apps/ML/learn/ray/serve/test/intent_prediction/intent2id.json'))
        self.id2intent = {v: k for k, v in self.label2id.items()}
        self.bert_model = BertForSequenceClassification.from_pretrained(
            "/home/ankur/dev/apps/ML/learn/ray/serve/test/intent_prediction")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    @serve.batch(max_batch_size=20, batch_wait_timeout_s=0.05)
    async def handle_batch(self, batch: List[str]) -> List[OutputModel]:
        batch_size = len(batch)
        logger.info(f"batch size = {batch_size}")
        PAD_LEN = 45
        token_dict = self.tokenizer.batch_encode_plus(
            batch, add_special_tokens=True, max_length=PAD_LEN, pad_to_max_length=True,
            truncation=True, return_attention_mask=True, return_tensors='pt')
        input_ids = token_dict['input_ids']
        attention_masks = token_dict['attention_mask']
        start = time.time()
        logits = self.bert_model(input_ids,
                                 token_type_ids=None,
                                 attention_mask=attention_masks,
                                 return_dict=False)[0]
        inference_time = time.time() - start
        logits = logits.detach().cpu().numpy()
        row_norms = np.linalg.norm(logits, axis=1, keepdims=True)
        logits = logits / row_norms
        # Take argmax along columns. Dimension of intent_id: (Batch, 1)
        intent_ids = np.argmax(logits, axis=1)
        intent_labels = [self.id2intent.get(k) for k in intent_ids]
        results = [OutputModel(prediction=label, scores=logits[idx], inference_time=inference_time) for idx, label in
                   enumerate(intent_labels)]
        return results


    # Here, In this example, InputModel and OutputModel define the expected structure and data types for the request and response.
    # FastAPI, integrated with Ray Serve through the @serve.ingress decorator, automatically validates incoming requests against the InputModel schema.
    # If the request data doesn't conform to the schema, FastAPI will return an error response, ensuring that the process method only receives valid data
    @app.post("/intent")
    async def process(self, input_data: InputModel) -> OutputModel:
        # limit number of threads
        num_threads = torch.get_num_threads()
        logger.debug("num_threads: {}".format(num_threads))
        ret = await self.handle_batch(input_data.text)
        return ret



intent_app = DetermineIntent.bind()
serve.start(
    http_options={
        "host": "0.0.0.0",
        "port": 8000,
        "request_timeout_s": 10,
        "keep_alive_timeout_s": 5
    }
)
# 2: Deploy the deployment.
serve.run(intent_app, route_prefix="/")

run_in_debugger = 1
if run_in_debugger:
    while 1:
        time.sleep(100)