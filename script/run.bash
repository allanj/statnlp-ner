#!/bin/bash

java -cp statnlp-ner-1.0.jar org.statnlp.example.linear_ne.EMainCRF --train_num 1000 \
     --test_num 1000 -it 4000 -t 40 --useEmb true --useDiscrete false > logs/emb_only.log 2>&1 

