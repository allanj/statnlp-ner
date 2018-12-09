package org.statnlp.example.linear_ne.character;

import java.util.ArrayList;
import java.util.List;

import org.statnlp.example.linear_ne.EInst;
import org.statnlp.example.linear_ne.character.CharacterNetworkCompiler.NodeType;
import org.statnlp.hypergraph.FeatureArray;
import org.statnlp.hypergraph.FeatureManager;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkIDMapper;


public class CharacterFeatureManager extends FeatureManager {

	private static final long serialVersionUID = 376931974939202432L;

	public enum FeaType{word, transition, char_emit, prefix, suffix};
	
	List<String> labels;
	boolean useCharFeats;
	public CharacterFeatureManager(GlobalNetworkParam param_g, List<String> labels, boolean useCharFeats) {
		super(param_g);
		this.labels = labels;
		this.useCharFeats = useCharFeats;
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
		CharacterNetworkCompiler.NodeType nodeType = NodeType.values()[nodeArr[3]];
		if (nodeType == NodeType.Leaf) return FeatureArray.EMPTY;
		if (nodeType == NodeType.Root && word_pos != inst.size() - 1) return FeatureArray.EMPTY;
		String entity = nodeType == NodeType.Root ? "ROOT" : this.labels.get(eId) ; 
//		System.out.println(network.getInstance().isLabeled() + " " + Arrays.toString(nodeArr));
		int[] child = NetworkIDMapper.toHybridNodeArray(network.getNode(children_k[0]));
		int childEId = child[2];
//		int prev_word_pos = child[0];
		
		
		String prevEntity = child[3] == NodeType.Leaf.ordinal() ? "LEAF" : this.labels.get(childEId);
//		if (prev_word_pos == word_pos) {
//			
//		} else {
//			String label = entity.length() > 2 ? entity.substring(0, entity.length() - 2) : entity;
//			String prevLabel = prevEntity.length() > 2 ? prevEntity.substring(0, prevEntity.length() - 2) : prevEntity;
//			featureList.add(this._param_g.toFeature(network, FeaType.transition.name(), label,  prevLabel));
//		}
		
		featureList.add(this._param_g.toFeature(network, FeaType.transition.name(), prevEntity,  entity));
		
		
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
//			entity = entity.length() > 2 ? entity.substring(0, entity.length() - 2) : entity;
			fs[0] = this._param_g.toFeature(network,FeaType.word.name() , entity, word);
			curr = curr.addNext(this.createFeatureArray(network, fs));
			
			if (this.useCharFeats) {
				List<Integer> fs2 =new ArrayList<>();
				for (int len = 1; len<=5;len++) {
					if (word.length()>=len) {
						fs2.add(this._param_g.toFeature(network, FeaType.prefix.name() +"-"+len, entity, word.substring(0, len)));
						fs2.add(this._param_g.toFeature(network, FeaType.suffix.name() +"-"+len, entity, word.substring(word.length() - len, word.length())));
					}
				}
				curr = curr.addNext(this.createFeatureArray(network, fs2));
			}
		}
		
		return fa;
	}
}
