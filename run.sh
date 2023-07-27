python3 main.py --plm='roberta-large' --type='ptuningv1'
python3 main.py --plm='roberta-large' --type='ptuningv2'
python3 main.py --plm='roberta-large' --type='lora'
python3 main.py --plm='bert' --type='lora' --train_file='datasets/preprocessed/SBERT/SBERT_train_th08.json' --valid_file='datasets/preprocessed/SBERT/SBERT_dev_th08.json' --test_file='datasets/preprocessed/SBERT/SBERT_test_th08.json' > results/230621/SBERT/SBERT_th08_LORA_1.log
