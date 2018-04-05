package org.statnlp.example.linear_ne;

import java.util.List;

import org.statnlp.commons.types.Sentence;
import org.statnlp.example.base.BaseInstance;


public class EInst extends BaseInstance<EInst, Sentence, List<String>> {

	private static final long serialVersionUID = 1851514046050983662L;
	
	public EInst(int instanceId, double weight) {
		super(instanceId, weight);
	}
	
	public EInst(int instanceId, double weight, Sentence sent) {
		super(instanceId, weight);
		this.input = sent;
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
}
