# LL-tJKdMmR3KcN7Jd3h5w4j9maubnrlfGRRXHgGfX3xXyWItM1evNYtdhZrOadxlpfD

from llamaapi import LlamaAPI
import json
import csv

l = LlamaAPI('LL-tJKdMmR3KcN7Jd3h5w4j9maubnrlfGRRXHgGfX3xXyWItM1evNYtdhZrOadxlpfD')


with open('sample.txt', 'r') as file:

    txt = csv.reader(file, delimiter='\t')
    next(txt)
    rows=[]
    
    for row in txt:
        word1 = row[0]
        word2 = row[1]

        api_request_json = {
        "model": "llama-13b-chat",
        "messages": [
            {"role": "system", "content": "Calculate the cosine similarity of the 2 words"},
            {"role": "user", "content": f"{word1} {word2}"},
        ]
        }

        # Run llama
        response = l.run(api_request_json)
        x = response.json()
        # print(x['content'])
        content = x['choices'][0]['message']['content']
        print(content)