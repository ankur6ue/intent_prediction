from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
import os
import ray
import tempfile
from utils import create_s3_client, boto_response_ok
from pathlib import Path
import logging
import time
import shutil


CACHE_DIR_ROOT = 'app_cache'

@ray.remote(num_returns=2)
def download_model(model_name, bucket_name, cache_location=True):
    # Note: When using ray, all of the tasks and actors are executed remotely in Ray’s worker processes.
    # Since Python logger module creates a singleton logger per process, loggers should be configured on
    # per task/actor basis.
    logging.basicConfig(level=logging.INFO)
    logging.info("reading data about model {0} from bucket {1}".format(model_name, bucket_name))
    model_data, s3_download_time = download_model_impl(model_name, bucket_name, logging, cache_location)
    return model_data, s3_download_time


def download_model_impl(model_name, bucket_name, logging, cache_location=True):
    # Logic: first see if we previously downloaded the model to a temporary directory. If so, a file in .cache directory
    # will have the location of the model. If not, download the model from S3 to a temporary directory, add the
    # temporary directory path to the cache and load the model from the temporary directory.
    cache_dir = os.path.join(Path.home(), CACHE_DIR_ROOT, model_name)
    if os.path.exists(os.path.join(cache_dir, 'model.safetensors')):
        s3_download_time = 0
        logging.info('found previously downloaded model location {0} in the cache'.format(cache_dir))
    else:
        logging.info('no cached model found, attempting to load {0} from S3 bucket {1}'.format(model_name, bucket_name))
        s3_client = create_s3_client(logging)
        if s3_client:
            resp = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=model_name + '/model/', Delimiter="/", MaxKeys=100)
            if boto_response_ok(resp) and resp.get("Contents"):
                start = time.time()
                for item in resp["Contents"]:
                    _, filename = os.path.split(item['Key'])
                    dest_path = os.path.join(cache_dir, filename)

                    s3_client.download_file(bucket_name, item['Key'], dest_path)
                s3_download_time = time.time() - start
                logging.info("s3 download time: {0}".format(s3_download_time))

            else:
                raise ValueError("error downloading {0} from s3".format(model_name))
        else:
            raise ValueError("error creating S3 client while trying to download speech2text model")
    # Now read the contents of the speech2text model files as a binary stream and put them into a dictionary
    if os.path.exists(cache_dir):
        objs = {}
        objs['cache_dir'] = cache_dir
        # Iterate over all the files in directory
        for folder_name, subfolders, filenames in os.walk(cache_dir):
            for filename in filenames:
                # create complete filepath of file in directory
                filepath = os.path.join(folder_name, filename)
                with open(filepath, mode='rb') as file:  # b is important -> binary
                    fileContent = file.read()
                    objs[filename] = fileContent
        if not cache_location: # if we are not using caching, delete the temporary directory
        #    shutil.rmtree(dest_dir)
            pass
        return objs, s3_download_time

    else:
        raise ValueError("error loading model {0}".format(model_name))
