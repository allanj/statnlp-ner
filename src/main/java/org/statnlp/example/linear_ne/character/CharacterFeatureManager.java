package org.statnlp.example.linear_ne.character;

import java.util.ArrayList;

import org.statnlp.example.linear_ne.ECRFNetworkCompiler;
import org.statnlp.example.linear_ne.ECRFNetworkCompiler.NodeType;
import org.statnlp.example.linear_ne.EInst;
import org.statnlp.hypergraph.FeatureArray;
import org.statnlp.hypergraph.FeatureManager;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkIDMapper;

public class CharacterFeatureManager extends FeatureManager {

	private static final long serialVersionUID = 376931974939202432L;

	public enum FeaType{word, transition, char_emit};
	
	public CharacterFeatureManager(GlobalNetworkParam param_g) {
		super(param_g);
	}
	
	@Override
	protected FeatureArray extract_helper(Network network, int parent_k, int[] children_k, int children_k_index) {
		
		EInst inst = ((EInst)network.getInstance());
		long node = network.getNode(parent_k);
		int[] nodeArr = NetworkIDMapper.toHybridNodeArray(node);
		
		FeatureArray fa = null;
		ArrayList<Integer> featureList = new ArrayList<Integer>();
		
		int word_pos = nodeArr[0];
		int char_pos = nodeArr[1];
		int eId = nodeArr[2];
		ECRFNetworkCompiler.NodeType nodeType = NodeType.values()[nodeArr[2]];
		if (nodeType == NodeType.Leaf) return FeatureArray.EMPTY;
		if (nodeType == NodeType.Root && word_pos != inst.size() - 1) return FeatureArray.EMPTY;
		String entity = eId + ""; 
		int[] child = NetworkIDMapper.toHybridNodeArray(network.getNode(children_k[0]));
		int childEId = child[1];
		
		String prevEntity =  childEId + "";
		featureList.add(this._param_g.toFeature(network, FeaType.transition.name(), entity,  prevEntity));
		
		if (nodeType == NodeType.Root) {
			return this.createFeatureArray(network, featureList);
		}
		
		String word = inst.getInput().get(word_pos).getForm();
		String char_form = word.substring(char_pos, char_pos + 1);
		fa = this.createFeatureArray(network, featureList);
		FeatureArray curr = fa;
		int[] fs0 = new int[1];
		fs0[0] = this._param_g.toFeature(network,FeaType.char_emit.name() , entity, char_form);
		curr = curr.addNext(this.createFeatureArray(network, fs0));
		
		if (char_pos == word.length() - 1) {
			int[] fs = new int[1];
			fs[0] = this._param_g.toFeature(network,FeaType.word.name() , entity, word);
			curr = curr.addNext(this.createFeatureArray(network, fs));
		}
		
		return fa;
	}
}
