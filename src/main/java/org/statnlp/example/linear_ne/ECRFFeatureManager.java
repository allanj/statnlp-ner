package org.statnlp.example.linear_ne;

import java.util.AbstractMap.SimpleImmutableEntry;

import org.statnlp.commons.types.Sentence;
import org.statnlp.example.linear_ne.ECRFNetworkCompiler.NodeType;
import org.statnlp.hypergraph.FeatureArray;
import org.statnlp.hypergraph.FeatureManager;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkConfig;
import org.statnlp.hypergraph.NetworkIDMapper;
import org.statnlp.hypergraph.neural.MultiLayerPerceptron;

import java.util.ArrayList;

public class ECRFFeatureManager extends FeatureManager {

	private static final long serialVersionUID = 376931974939202432L;

	public enum FeaType{
		word, tag, lw, lt, ltt, rw, rt, prefix, suffix,
		transition};
	private String OUT_SEP = MultiLayerPerceptron.OUT_SEP; 
	private String IN_SEP = MultiLayerPerceptron.IN_SEP;
	private final String START = "STR";
	private final String END = "END";
	
	private boolean lowercase = true;
	private Entity[] labels;
	
	public ECRFFeatureManager(GlobalNetworkParam param_g, Entity[] labels, String neuralType, boolean moreBinaryFeatures, boolean lowercase) {
		super(param_g);
		this.labels = labels;
		this.lowercase = lowercase;
	}
	
	@Override
	protected FeatureArray extract_helper(Network network, int parent_k, int[] children_k, int children_k_index) {
		
		EInst inst = ((EInst)network.getInstance());
		Sentence sent = inst.getInput();
		long node = network.getNode(parent_k);
		int[] nodeArr = NetworkIDMapper.toHybridNodeArray(node);
		
		FeatureArray fa = null;
		ArrayList<Integer> featureList = new ArrayList<Integer>();
		
		int pos = nodeArr[0];
		int eId = nodeArr[1];
		ECRFNetworkCompiler.NodeType nodeType = NodeType.values()[nodeArr[2]];
		if (nodeType == NodeType.Leaf) return FeatureArray.EMPTY;
		if (nodeType == NodeType.Root && pos != inst.size() - 1) return FeatureArray.EMPTY;
		String entity = eId + ""; 
		int[] child = NetworkIDMapper.toHybridNodeArray(network.getNode(children_k[0]));
		int childEId = child[1];
		
		String prevEntity =  childEId + "";
		featureList.add(this._param_g.toFeature(network, FeaType.transition.name(), entity,  prevEntity));
		
		if (nodeType == NodeType.Root) {
			return this.createFeatureArray(network, featureList);
		}
		
		String word = pos != inst.size() ? inst.getInput().get(pos).getForm() : END;
		
		if(NetworkConfig.USE_NEURAL_FEATURES){
			if (eId != 0 && eId != this.labels.length - 1) {
				Object input = null;
				String sentenceInput = this.lowercase ? sent.toString().toLowerCase() : sent.toString();
				input = new SimpleImmutableEntry<String, Integer>(sentenceInput, pos);
				this.addNeural(network, 0, parent_k, children_k_index, input, eId - 1);
			}
		} else {
			featureList.add(this._param_g.toFeature(network,FeaType.word.name(), entity, word));
		}
		
		fa = this.createFeatureArray(network, featureList);
		return fa;
	}
}
