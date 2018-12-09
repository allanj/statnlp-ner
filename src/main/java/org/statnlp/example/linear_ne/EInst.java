package org.statnlp.example.linear_ne;

import java.util.List;

import org.statnlp.commons.types.Sentence;
import org.statnlp.example.base.BaseInstance;


public class EInst extends BaseInstance<EInst, Sentence, List<String>> {

	private static final long serialVersionUID = 1851514046050983662L;
	
	public List<List<String>> detailedCharacterPrediction;
	
	public EInst(int instanceId, double weight) {
		super(instanceId, weight);
	}
	
	public EInst(int instanceId, double weight, Sentence sent, List<String> output) {
		super(instanceId, weight);
		this.input = sent;
		this.output = output;
	}
	
	@Override
	public int size() {
		return this.input.length();
	}
	
	public Sentence duplicateInput(){
		return input;
	}
	
	public List<String> duplicateOutput() {
		return this.output;
	}

	public void setDetails(List<List<String>> detailedCharacterPrediction) {
		this.detailedCharacterPrediction = detailedCharacterPrediction;
	}
}
