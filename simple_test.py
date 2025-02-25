import requests

url = 'http://127.0.0.1:8000/intent'
myobj = {'text': "add this song to the album"}

x = requests.post(url, json = myobj)

print(x.text)