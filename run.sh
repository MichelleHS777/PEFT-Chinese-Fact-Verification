  python3 main.py --eval --plm='bert' --type='lora' --train_file='datasets/preprocessed/SBERT/SBERT_train_th08.json' --valid_file='datasets/preprocessed/SBERT/SBERT_dev_th08.json' --test_file='datasets/preprocessed/sent_nars_th08.json'
# python3 main2.py --eval --plm='bert' --type='lora' --train_file='datasets/preprocessed/SBERT/SBERT_train_th08.json' --valid_file='datasets/preprocessed/SBERT/SBERT_dev_th08.json' --test_file='datasets/preprocessed/SBERT/SBERT_test_th08.json' > label.log
# python3 main.py --plm='bert' --type='lora' --train_file='datasets/preprocessed/train.json' --valid_file='datasets/preprocessed/dev.json' --test_file='datasets/preprocessed/test.json' > results/230614/gold/bert/gold_LORA_3.log
# python3 main.py --plm='bert' --type='lora' --train_file='datasets/preprocessed/train.json' --valid_file='datasets/preprocessed/dev.json' --test_file='datasets/preprocessed/test.json' > results/230614/gold/bert/gold_LORA_4.log
# python3 main.py --plm='bert' --type='lora' --train_file='datasets/preprocessed/train.json' --valid_file='datasets/preprocessed/dev.json' --test_file='datasets/preprocessed/test.json' > results/230614/gold/bert/gold_LORA_5.log

#python3 main.py --plm='ernie' --type='lora' --train_file='datasets/preprocessed/train.json' --valid_file='datasets/preprocessed/dev.json' --test_file='datasets/preprocessed/test.json' > results/230614/gold/ernie/gold_LORA_2.log
#python3 main.py --plm='ernie' --type='lora' --train_file='datasets/preprocessed/train.json' --valid_file='datasets/preprocessed/dev.json' --test_file='datasets/preprocessed/test.json' > results/230614/gold/ernie/gold_LORA_3.log
#python3 main.py --plm='ernie' --type='lora' --train_file='datasets/preprocessed/train.json' --valid_file='datasets/preprocessed/dev.json' --test_file='datasets/preprocessed/test.json' > results/230614/gold/ernie/gold_LORA_4.log

##python3 main.py --plm='bert' --type='lora' --train_file='datasets/preprocessed/prompt/prompt_train_claim_sent_out5.json' --valid_file='datasets/preprocessed/prompt/prompt_dev_claim_sent_out5.json' --test_file='datasets/preprocessed/prompt/prompt_test_claim_sent_out5.json' > results/230614/prompt/bert/prompt_out5_LORA_1.log
#python3 main.py --plm='bert' --type='lora' --train_file='datasets/preprocessed/prompt/prompt_train_claim_sent_out5.json' --valid_file='datasets/preprocessed/prompt/prompt_dev_claim_sent_out5.json' --test_file='datasets/preprocessed/prompt/prompt_test_claim_sent_out5.json' > results/230614/prompt/bert/prompt_out5_LORA_2.log
##python3 main.py --plm='bert' --type='lora' --train_file='datasets/preprocessed/prompt/prompt_train_claim_sent_out5.json' --valid_file='datasets/preprocessed/prompt/prompt_dev_claim_sent_out5.json' --test_file='datasets/preprocessed/prompt/prompt_test_claim_sent_out5.json' > results/230614/prompt/bert/prompt_out5_LORA_3.log
#python3 main.py --plm='bert' --type='lora' --train_file='datasets/preprocessed/prompt/prompt_train_claim_sent_out5.json' --valid_file='datasets/preprocessed/prompt/prompt_dev_claim_sent_out5.json' --test_file='datasets/preprocessed/prompt/prompt_test_claim_sent_out5.json' > results/230614/prompt/bert/prompt_out5_LORA_4.log
#python3 main.py --plm='bert' --type='lora' --train_file='datasets/preprocessed/prompt/prompt_train_claim_sent_out5.json' --valid_file='datasets/preprocessed/prompt/prompt_dev_claim_sent_out5.json' --test_file='datasets/preprocessed/prompt/prompt_test_claim_sent_out5.json' > results/230614/prompt/bert/prompt_out5_LORA_5.log


#python3 main.py --plm='bert-large' --type='ptuningv1' > results/230614/ptuningv1/bertlarge_ptuningv1_1.log
#python3 main.py --plm='bert-large' --type='ptuningv1' > results/230614/ptuningv1/bertlarge_ptuningv1_2.log
#python3 main.py --plm='bert-large' --type='ptuningv1' > results/230614/ptuningv1/bertlarge_ptuningv1_3.log
#python3 main.py --plm='bert-large' --type='ptuningv1' > results/230614/ptuningv1/bertlarge_ptuningv1_4.log
#python3 main.py --plm='bert-large' --type='ptuningv1' > results/230614/ptuningv1/bertlarge_ptuningv1_5.log
#
#python3 main.py --plm='bert-large' --type='ptuningv2' > results/230614/ptuningv2/bertlarge_ptuningv2_1.log
#python3 main.py --plm='bert-large' --type='ptuningv2' > results/230614/ptuningv2/bertlarge_ptuningv2_2.log
#python3 main.py --plm='bert-large' --type='ptuningv2' > results/230614/ptuningv2/bertlarge_ptuningv2_3.log
#python3 main.py --plm='bert-large' --type='ptuningv2' > results/230614/ptuningv2/bertlarge_ptuningv2_4.log
#python3 main.py --plm='bert-large' --type='ptuningv2' > results/230614/ptuningv2/bertlarge_ptuningv2_5.log

# python3 main.py --plm='bert' --type='lora' --train_file='datasets/preprocessed/SBERT/SBERT_train_th08.json' --valid_file='datasets/preprocessed/SBERT/SBERT_dev_th08.json' --test_file='datasets/preprocessed/SBERT/SBERT_test_th08.json' > results/230614/SBERT/bert/SBERT_th08_LORA_1.log
# python3 main.py --plm='bert' --type='lora' --train_file='datasets/preprocessed/prompt/prompt_train_claim_sent_th08.json' --valid_file='datasets/preprocessed/prompt/prompt_dev_claim_sent_th08.json' --test_file='datasets/preprocessed/prompt/prompt_test_claim_sent_th08.json' > results/230614/prompt/bert/prompt_th08_LORA_1.log
# python3 main.py --plm='bert' --type='lora'  --train_file='datasets/preprocessed/hybrid/hybrid_train.json' --valid_file='datasets/preprocessed/hybrid/hybrid_dev.json' --test_file='datasets/preprocessed/hybrid/hybrid_test.json' > results/230614/hybrid/bert/hybrid_LORA_1.log
# python3 main.py --plm='bert' --type='lora' --train_file='datasets/preprocessed/surface/surface_train.json' --valid_file='datasets/preprocessed/surface/surface_dev.json' --test_file='datasets/preprocessed/surface/surface_test.json' > results/230614/surface/bert/surface_LORA_1.log
# python3 main.py --plm='bert' --type='lora'  --train_file='datasets/preprocessed/semantic/semantic_train.json' --valid_file='datasets/preprocessed/semantic/semantic_dev.json' --test_file='datasets/preprocessed/semantic/semantic_test.json' > results/230614/semantic/bert/semantic_LORA_1.log

# python3 main.py --plm='ernie' --type='lora' --train_file='datasets/preprocessed/SBERT/SBERT_train_th08.json' --valid_file='datasets/preprocessed/SBERT/SBERT_dev_th08.json' --test_file='datasets/preprocessed/SBERT/SBERT_test_th08.json' > results/230614/SBERT/ernie/SBERT_th08_LORA_1.log
# python3 main.py --plm='ernie' --type='lora' --train_file='datasets/preprocessed/prompt/prompt_train_claim_sent_th08.json' --valid_file='datasets/preprocessed/prompt/prompt_dev_claim_sent_th08.json' --test_file='datasets/preprocessed/prompt/prompt_test_claim_sent_th08.json' > results/230614/prompt/ernie/prompt_th08_LORA_1.log
# python3 main.py --plm='ernie' --type='lora'  --train_file='datasets/preprocessed/hybrid/hybrid_train.json' --valid_file='datasets/preprocessed/hybrid/hybrid_dev.json' --test_file='datasets/preprocessed/hybrid/hybrid_test.json' > results/230614/hybrid/ernie/hybrid_LORA_1.log
# python3 main.py --plm='ernie' --type='lora' --train_file='datasets/preprocessed/surface/surface_train.json' --valid_file='datasets/preprocessed/surface/surface_dev.json' --test_file='datasets/preprocessed/surface/surface_test.json' > results/230614/surface/ernie/surface_LORA_1.log
# python3 main.py --plm='ernie' --type='lora'  --train_file='datasets/preprocessed/semantic/semantic_train.json' --valid_file='datasets/preprocessed/semantic/semantic_dev.json' --test_file='datasets/preprocessed/semantic/semantic_test.json' > results/230614/semantic/ernie/semantic_LORA_1.log

#python3 main.py --plm='roberta-large' --type='ptuningv1' > results/230606/ptuningv1/roberta/robertalarge_ptuningv1_1.log
#python3 main.py --plm='roberta-large' --type='ptuningv2' > results/230606/ptuningv2/roberta/robertalarge_ptuningv2_1.log
#
#python3 main.py --plm='roberta-large' --rank=4 --type='lora' > results/230606/lora/roberta/robertalarge_lora_rank4_qkvdense2.log
#python3 main.py --plm='roberta-large' --rank=4 --type='lora' > results/230606/lora/roberta/robertalarge_lora_rank4_qkvdense3.log
#python3 main.py --plm='roberta-large' --rank=4 --type='lora' > results/230606/lora/roberta/robertalarge_lora_rank4_qkvdense4.log
#python3 main.py --plm='roberta-large' --rank=4 --type='lora' > results/230606/lora/roberta/robertalarge_lora_rank4_qkvdense5.log
#
#python3 main.py --plm='roberta-large' --type='ptuningv1' > results/230606/ptuningv1/roberta/robertalarge_ptuningv1_2.log
#python3 main.py --plm='roberta-large' --type='ptuningv1' > results/230606/ptuningv1/roberta/robertalarge_ptuningv1_3.log
#python3 main.py --plm='roberta-large' --type='ptuningv1' > results/230606/ptuningv1/roberta/robertalarge_ptuningv1_4.log
#python3 main.py --plm='roberta-large' --type='ptuningv1' > results/230606/ptuningv1/roberta/robertalarge_ptuningv1_5.log
#
#python3 main.py --plm='roberta-large' --type='ptuningv2' > results/230606/ptuningv2/roberta/robertalarge_ptuningv2_2.log
#python3 main.py --plm='roberta-large' --type='ptuningv2' > results/230606/ptuningv2/roberta/robertalarge_ptuningv2_3.log
#python3 main.py --plm='roberta-large' --type='ptuningv2' > results/230606/ptuningv2/roberta/robertalarge_ptuningv2_4.log
#python3 main.py --plm='roberta-large' --type='ptuningv2' > results/230606/ptuningv2/roberta/robertalarge_ptuningv2_5.log
