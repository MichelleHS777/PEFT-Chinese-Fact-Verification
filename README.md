# PEFT-Chinese-Fact-Verification
## Basic Usage 
If you want to implement the code, you need to define:  
`--plm`: bert, bert-large, roberta, ernie, roberta-large  
`--type`: ptuningv1, ptuningv2, lora  
`--train`: Set if train    
`--eval`: Set if evaluate   
`--train_file` (Eg: 'datasets/preprocessed/train.json')  
`--valid_file` (Eg: 'datasets/preprocessed/dev.json')  
`--test_file` (Eg: 'datasets/preprocessed/test.json')

## Preprocessing
python preprocess.py --input_path="/path/to/your/html/folder/" --output_path="/path/to/your/output/folder/"
