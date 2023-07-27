from tqdm import tqdm
import json

dataset = open('./datasets/preprocessed/sent_nars_th08.json', 'r', encoding='utf-8')
save_file = open('./datasets/preprocessed/sent_nars_th082.json', 'w', encoding='utf-8')

for data in tqdm(dataset, desc='Preprocess...'):
    data = eval(data)
    # claimId = data['claimId']
    claim = data['claim']
    evidences = data['evidences']
    evidences = ''.join(evidences)
    # label = data['label']
    data = json.dumps({'claim': claim, 'evidences': evidences}, ensure_ascii=False)
    # data = json.dumps({'claimId': int(claimId), 'claim': claim, 'evidences': evidences, 'label': label}, ensure_ascii=False)
    save_file.write(data + "\n")
save_file.close()
