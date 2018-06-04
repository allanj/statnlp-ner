#!/bin/bash


java -cp statnlp-ner-1.0.jar org.statnlp.example.linear_ne.EMain --train_num 500 --dev_num 100 \
  --test_num 100 --batch true 1 --optimizer sgdclip  \
  -lstm true -it 50000 --evalDev true 250 \
  -emb random -os linux -l2 0 > logs/baseline.log 2>&1 

