import json

with open('/home/namd/RAMSEDataset/data/train.jsonlines') as f:
    lines = f.read().split('\n')

for line in lines:
    line = line.strip()
    doc = json.loads(line)
    print(doc['sentence_data'])
    break