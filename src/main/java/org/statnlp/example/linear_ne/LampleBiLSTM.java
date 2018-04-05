package org.statnlp.example.linear_ne;

import java.util.AbstractMap.SimpleImmutableEntry;

import org.statnlp.hypergraph.neural.NeuralNetworkCore;

public class LampleBiLSTM extends NeuralNetworkCore {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6175132248082165824L;

	public LampleBiLSTM(String className, 
			int hiddenSize, boolean bidirection, String optimizer, double learningRate, int clipping, int numLabels, int gpuId, String embedding, double clipvalue) {
		super(numLabels, gpuId);
		config.put("class", className);
        config.put("hiddenSize", hiddenSize);
        config.put("bidirection", bidirection);
        config.put("optimizer", optimizer);
        config.put("numLabels", numLabels);
        config.put("embedding", embedding);
        config.put("learningRate", learningRate);
        config.put("clipping", clipvalue);
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
