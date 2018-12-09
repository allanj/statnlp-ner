package org.statnlp.example.linear_ne;

import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.ArrayList;
import java.util.List;

import org.statnlp.commons.types.Sentence;
import org.statnlp.example.linear_ne.ECRFNetworkCompiler.NodeType;
import org.statnlp.hypergraph.FeatureArray;
import org.statnlp.hypergraph.FeatureManager;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkConfig;
import org.statnlp.hypergraph.NetworkIDMapper;

public class ECRFFeatureManager extends FeatureManager {

	private static final long serialVersionUID = 376931974939202432L;

	public enum FeaType{
		word, transition, prefix, suffix};
	
	boolean useCharFeats;
	
	public ECRFFeatureManager(GlobalNetworkParam param_g, boolean useCharFeats) {
		super(param_g);
		this.useCharFeats = useCharFeats;
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
		
//		String word = inst.getInput().get(pos).getForm();
		fa = this.createFeatureArray(network, featureList);
		FeatureArray curr = fa;
		if(NetworkConfig.USE_NEURAL_FEATURES){
			Object input = null;
			String sentenceInput = sent.toString();
			input = new SimpleImmutableEntry<String, Integer>(sentenceInput, pos);
			this.addNeural(network, 0, parent_k, children_k_index, input, eId);
		} else {
			String word = sent.get(pos).getForm();
			int[] fs0 = new int[1];
			fs0[0] = this._param_g.toFeature(network,FeaType.word.name(), entity, word);
			curr = curr.addNext(this.createFeatureArray(network, fs0));
			if (this.useCharFeats) {
				List<Integer> fs =new ArrayList<>();
				for (int len = 1; len<=5;len++) {
					if (word.length()>=len) {
						fs.add(this._param_g.toFeature(network, FeaType.prefix.name() +"-"+len, entity, word.substring(0, len)));
						fs.add(this._param_g.toFeature(network, FeaType.suffix.name() +"-"+len, entity, word.substring(word.length() - len, word.length())));
					}
				}
				curr = curr.addNext(this.createFeatureArray(network, fs));
			}
		}
		
		return fa;
	}
}
