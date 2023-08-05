# PEFT-Chinese-Fact-Verification
## Claim Verification
* P-Tuning
* P-Tuning v2
* LoRA (rank=4 is the best)
## Template Engineering
* We only define 10 soft tokens for peft
## Preprocess
    python preprocess.py 
`--dataset` choose the dataset path you want to preprocess (default='datasets/unpreprocess/train.json')   
`--save_file` save the preprocess file (default='datasets/preprocessed/train.json')  
## Basic Usage  
    python main.py --train --eval
## Arguments  
`--plm` bert, bert-large, roberta, ernie, roberta-large  
`--type` ptuningv1, ptuningv2, lora  
`--train` Remember to set this if you want to train.    
`--eval` Remember to set this if you want to evaluate.   
`--train_file` (Eg: 'datasets/preprocessed/train.json')  
`--valid_file` (Eg: 'datasets/preprocessed/dev.json')  
`--test_file` (Eg: 'datasets/preprocessed/test.json')
