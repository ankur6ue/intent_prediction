from locust import HttpUser, task, events, constant_throughput
from locust.env import Environment
import gevent
from locust.log import setup_logging
from locust.stats import stats_history, stats_printer
import random
import requests
import json

# url = "http://192.168.1.168:8000/intent"
# data = {"text": "add this song to the album"}

# response = requests.post(url, data=json.dumps(data), timeout=2)

# print(response.status_code)

setup_logging("INFO")
host = "http://192.168.1.168:8000/intent"
def load_snips_file(file_path):
    list_pair =[]
    with open(file_path,'r',encoding="utf8") as f:
        for line in f:
            split_line = line.split('\t')
            pair = split_line[0],split_line[1]
            list_pair.append(pair)
    return list_pair

TEST_PATH='test.tsv'
test_examples = load_snips_file(TEST_PATH)
class HelloWorldUser(HttpUser):
    host = "http://192.168.1.168:8000/intent"
    @task
    def hello_world(self):
        indx = (int)((len(test_examples) - 1) * random.random())
        example = test_examples[indx]
        data = {'text': example[1]}
        response = self.client.post(host, json=data)
        # response = self.client.post(host, data={'body': 'data'})
        ret = response.json()
        correct = ret['prediction'] == example[0]
        print("prediction = %s, model execution time = %0.3f, result correct: %s"
              % (ret['prediction'], ret["inference_time"], correct))
    # Each user will send 4 request every sec
    wait_time = constant_throughput(4)

# if launched directly, e.g. "python3 debugging.py", not "locust -f debugging.py"
if __name__ == "__main__":
    # setup Environment and Runner
    env = Environment(user_classes=[HelloWorldUser], events=events)
    runner = env.create_local_runner()
    gevent.spawn(stats_printer(env.stats))
    gevent.spawn(stats_history, env.runner)
    runner.start(40, spawn_rate=4)
    # gevent.spawn(stats_history(env.stats))
    # in 30 seconds stop the runner
    gevent.spawn_later(40, runner.quit)

    # wait for the greenlets
    runner.greenlet.join()