
## StatNLP Neural Named Entity Recognition System
We implemented an named entity recognition system using the StatNLP structure framework. 
The implementation largely follows Lample et al., 2016. 

## Benchmark Comparison
1. Simple Scenario

Model|\#Train | \#Dev | \#Tests
------------|-------| ------------- | -----
BiGRU-CRF |500| 100 | 100
Lample LSTM-CRF|500 | 100 | 100


## References
Lample, Guillaume, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, and Chris Dyer. "Neural Architectures for Named Entity Recognition." In *Proceedings of NAACL-HLT*, pp. 260-270. 2016.

## TODO
1. Make a faster version to save the model in the middle of training
2. Faster version of LSTM/GRU
3. Add CNN in Torch for character-level embedding
4. More experiments