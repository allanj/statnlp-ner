#!/bin/bash




java -cp statnlp-ner-1.0.jar org.statnlp.example.linear_ne.EMainCRF --train_num -1 \
     --test_num -1 -it 4000 -t 40 --useEmb true --useDiscrete false --useBigram true --wordHalfWindow 1  > logs/emb_2gram_window1.log 2>&1 

java -cp statnlp-ner-1.0.jar org.statnlp.example.linear_ne.EMainCRF --train_num -1 \
     --test_num -1 -it 4000 -t 40 --useEmb true --useDiscrete false --useBigram true --wordHalfWindow 2  > logs/emb_2gram_window2.log 2>&1 
.
