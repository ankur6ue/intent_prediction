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
from dotenv import load_dotenv, dotenv_values
from download_model import download_model_impl, download_model
import os
import ray
from pathlib import Path

logger = logging.getLogger("ray.serve")


class InputModel(BaseModel):
    text: str

class OutputModel(BaseModel):
    prediction: str
    scores: List[float]
    inference_time: float

app = FastAPI()
@serve.deployment(
    num_replicas="auto",
    autoscaling_config={"min_replicas": 1,
                        "upscale_delay_s": 10,
                        "downscale_delay_s": 5,
                        "metrics_interval_s": 5,
                        "max_replicas": 5},
    ray_actor_options={"num_cpus": 2, "num_gpus": 0})
@serve.ingress(app)
class DetermineIntent:
    def __init__(self, config):
        # limit number of threads.. no need to do the below manually. It seems to be done either by pytorch or ray
        # based on the num_cpus in the serve.deployment configuration
        # num_threads = 5
        # torch.set_num_threads(num_threads)
        # torch.set_num_interop_threads(num_threads)
        # Load model

        bucket_name = os.getenv("S3_BUCKET_NAME")
        model_name = "intent_detection"
        CACHE_DIR_ROOT = 'app_cache'
        cache_dir = os.path.join(Path.home(), CACHE_DIR_ROOT, model_name)
        if not os.path.exists(os.path.join(cache_dir, 'model.safetensors')):
            logging.info("reading data about model {0} from bucket {1}".format(model_name, bucket_name))
            remote_obj1, remote_obj2 = download_model.remote(model_name, bucket_name, logging)
            objs = ray.get(remote_obj1)
            s3_download_time = ray.get(remote_obj2)
            for k, v in objs.items():
                if k != 'cache_dir':
                    filepath = os.path.join(cache_dir, k)
                    with open(filepath, mode='wb') as file:  # b is important -> binary
                        file.write(v)

        self.label2id = json.load(open(os.path.join(cache_dir, 'intent2id.json')))
        logger.debug("loading the model..")
        # self.label2id = json.load(open('/home/ankur/dev/apps/ML/learn/ray/serve/test/intent_prediction/intent2id.json'))
        self.id2intent = {v: k for k, v in self.label2id.items()}
        self.bert_model = BertForSequenceClassification.from_pretrained(cache_dir)
        self.tokenizer = BertTokenizer.from_pretrained(cache_dir, do_lower_case=True)

    # Here, In this example, InputModel and OutputModel define the expected structure and data types for the request and response.
    # FastAPI, integrated with Ray Serve through the @serve.ingress decorator, automatically validates incoming requests against the InputModel schema.
    # If the request data doesn't conform to the schema, FastAPI will return an error response, ensuring that the process method only receives valid data
    @app.post("/intent")
    def process(self, input_data: InputModel) -> OutputModel:
        # limit number of threads
        num_threads = torch.get_num_threads()
        logger.debug("num_threads: {}".format(num_threads))
        logger.debug("Handling intent detection request on str {%s}", input_data.text)
        PAD_LEN = 45
        token_dict = self.tokenizer.encode_plus(
            input_data.text, add_special_tokens=True, max_length=PAD_LEN, pad_to_max_length=True,
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
        # logits is a batch, so take the first element
        logits_0 = logits[0]
        intent_id = np.argmax(logits_0).flatten()
        intent_label = self.id2intent.get(intent_id[0])
        logger.debug("Detected intent: %s" % intent_label )
        return {'prediction': intent_label, 'scores': logits_0.tolist(), 'inference_time': inference_time}



config = dotenv_values(".env")
# set environment variables from config.env
for k, v in config.items():
    os.environ[k] = v
env = os.getenv("ENV")

if env == "local":
    # create a local ray cluster
    resources = {'num_cpus': 20}
    ray.init(resources=resources)
else:
    ray.init(address=ray_cluster_url,
             runtime_env={"working_dir": working_dir,
                          "excludes": ["/models/*", "*.flac", "/.git/*"],
                          "env_vars": config})

serve.start(
    http_options={
        "host": "0.0.0.0",
        "port": 8000,
        "request_timeout_s": 10,
        "keep_alive_timeout_s": 5
    }
)
intent_app = DetermineIntent.bind(config)

# 2: Deploy the deployment.
serve.run(intent_app, route_prefix="/")

run_in_debugger = 1
if run_in_debugger:
    while 1:
        time.sleep(100)


