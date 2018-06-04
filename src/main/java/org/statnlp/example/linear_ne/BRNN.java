package org.statnlp.example.linear_ne;

import java.util.AbstractMap.SimpleImmutableEntry;

import org.statnlp.hypergraph.neural.NeuralNetworkCore;

public class BRNN extends NeuralNetworkCore {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6175132248082165824L;

	public BRNN(String className, int numLabels, int gpuId, String embedding,
			boolean fixEmbedding, double dropout, int hiddenSize, int embeddingSize) {
		super(numLabels, gpuId);
		config.put("class", className);
        config.put("numLabels", numLabels);
        config.put("embedding", embedding);
        config.put("hiddenSize", hiddenSize);
        config.put("embeddingSize", embeddingSize);
	}
	
	@Override
	public int hyperEdgeInput2OutputRowIndex(Object edgeInput) {
		@SuppressWarnings("unchecked")
		SimpleImmutableEntry<String, Integer> sentAndPos = (SimpleImmutableEntry<String, Integer>) edgeInput;
		int sentID = this.getNNInputID(sentAndPos.getKey());
		int row = sentAndPos.getValue() * this.getNNInputSize() + sentID;
		return row;
	}

	@Override
	public Object hyperEdgeInput2NNInput(Object edgeInput) {
		@SuppressWarnings("unchecked")
		SimpleImmutableEntry<String, Integer> sentAndPos = (SimpleImmutableEntry<String, Integer>) edgeInput;
		return sentAndPos.getKey();
	}

}
