# PEFT-Chinese-Fact-Verification
## Basic Usage  
    python main.py  
## Arguments  
`--plm` bert, bert-large, roberta, ernie, roberta-large  
`--type` ptuningv1, ptuningv2, lora  
`--train` Remember to set this if you want to train.    
`--eval` Remember to set this if you want to evaluate.   
`--train_file` (Eg: 'datasets/preprocessed/train.json')  
`--valid_file` (Eg: 'datasets/preprocessed/dev.json')  
`--test_file` (Eg: 'datasets/preprocessed/test.json')
