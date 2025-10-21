import json

with open('../data/train_data1.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    hin_train = data['English-Hindi']['Train']
    ben_train = data['English-Bengali']['Train']
    
    with open('../data/merged.txt', 'w', encoding='utf-8') as outfile:
        for example in hin_train.values():
            outfile.write(example.get('source') + '\n')
            outfile.write(example.get('target') + '\n')
        
        for example in hin_train.values():
            outfile.write(example.get('source') + '\n')
            outfile.write(example.get('target') + '\n')
