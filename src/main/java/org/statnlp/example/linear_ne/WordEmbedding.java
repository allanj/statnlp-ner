package org.statnlp.example.linear_ne;

public interface WordEmbedding {
	
	void readEmbedding(String file);
	
	double[] getEmbedding(String word);
	
	void clearEmbeddingMemory();
	
	int getDimension();
}
