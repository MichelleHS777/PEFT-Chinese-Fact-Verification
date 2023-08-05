python3 main.py --plm='roberta-large' --type='ptuningv1' > results_pt1.log
python3 main.py --plm='roberta-large' --type='ptuningv2' > results_pt2.log
python3 main.py --plm='roberta-large' --type='lora' > results_lora.log
python3 main.py --plm='bert' --type='lora' --train_file='datasets/preprocessed/SBERT/SBERT_train_th08.json' --valid_file='datasets/preprocessed/SBERT/SBERT_dev_th08.json' --test_file='datasets/preprocessed/SBERT/SBERT_test_th08.json' > SBERT_th08_LORA.log
