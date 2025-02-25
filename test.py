import asyncio
import logging
import time
from fastapi import FastAPI
from ray import serve
from ray.serve import batch
import logging
from typing import List
logger = logging.getLogger("ray.serve")
import numpy as np
from pydantic import BaseModel
from ray import serve
from ray.serve.handle import DeploymentHandle
from download_model import download_model_impl
import os
from dotenv import dotenv_values
from pathlib import Path
import json
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
logging.basicConfig(level=logging.INFO)



if __name__ == "__main__":
    config = dotenv_values(".env")
    # set environment variables from config.env
    for k, v in config.items():
        os.environ[k] = v
    bucket_name = os.getenv("S3_BUCKET_NAME")
    model_name = "intent_detection"
    logging.info("reading data about model {0} from bucket {1}".format(model_name, bucket_name))
    objs, s3_read_time = download_model_impl(model_name, bucket_name, logging)
    CACHE_DIR_ROOT = 'app_cache'
    cache_dir = os.path.join(Path.home(), CACHE_DIR_ROOT, model_name)
    for k, v in objs[0].items():
        if k is not 'cache_dir':
            filepath = os.path.join(cache_dir, k)
            with open(filepath, mode='wb') as file:  # b is important -> binary
                file.write(v)

    label2id = json.load(open(os.path.join(cache_dir, 'intent2id.json')))
    id2intent = {v: k for k, v in label2id.items()}
    bert_model = BertForSequenceClassification.from_pretrained(cache_dir)
    tokenizer = BertTokenizer.from_pretrained(cache_dir, do_lower_case=True)

    app = FastAPI()

    class InputModel(BaseModel):
        text: float

    @serve.deployment
    @serve.ingress(app)
    class TestApp:
        def __init__(self):
            logger.debug("loading the model..")
            pass

        @batch(max_batch_size=5, batch_wait_timeout_s=0.2)
        async def handle_batch(self, requests: List[float]) -> List[str]:
            num_requests = len(requests)
            logger.info(f"received {num_requests} requests")
            await asyncio.sleep(0.1)  # Simulate processing time
            return [f"processed {request: 0.2f}" for request in requests]
        @app.post("/intent")
        # @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.1)
        async def process(self, input_data: InputModel) -> str:
            # Use numpy's vectorized computation to efficiently process a batch.
            ret = await self.handle_batch(input_data.text)
            return ret



    tapp = TestApp.bind()
    serve.start(
        http_options={
            "host": "0.0.0.0",
            "port": 8000,
            "request_timeout_s": 10,
            "keep_alive_timeout_s": 5
        }
    )
    # 2: Deploy the deployment.
    serve.run(tapp, route_prefix="/")

    run_in_debugger = 1
    if run_in_debugger:
        while 1:
            time.sleep(100)