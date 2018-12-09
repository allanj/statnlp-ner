package org.statnlp.example.linear_ne.character;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.statnlp.commons.types.Instance;
import org.statnlp.commons.types.Sentence;
import org.statnlp.example.base.BaseNetwork;
import org.statnlp.example.base.BaseNetwork.NetworkBuilder;
import org.statnlp.example.linear_ne.EConf;
import org.statnlp.example.linear_ne.EInst;
import org.statnlp.hypergraph.LocalNetworkParam;
import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkCompiler;
import org.statnlp.hypergraph.NetworkIDMapper;

public class CharacterNetworkCompiler extends NetworkCompiler{

	private static final long serialVersionUID = -2388666010977956073L;

	public enum NodeType {Leaf, Node, Root};
	public static final int _size = 1300;
	//0: should be start tag
	//length -1: should be end tag;
	private List<String> labels;
	private Map<String, Integer> labelIndex;
	private final boolean DEBUG = true;
	
	static {
		NetworkIDMapper.setCapacity(new int[]{_size, _size, 80, 3});
	}
	
	public CharacterNetworkCompiler(List<String> labels){
		this.labels = labels;
		this.labelIndex = new HashMap<>(this.labels.size());
		for (int l = 0; l < this.labels.size(); l++) {
			this.labelIndex.put(this.labels.get(l), l);
		}
		System.out.println("number of labels: " + this.labels.size());
		System.out.println(this.labelIndex);
	}
	
	public long toNode_leaf(){
		//since 0 is the start_tag index;
		int[] arr = new int[]{0, 0, 0, NodeType.Leaf.ordinal()};
		return NetworkIDMapper.toHybridNodeID(arr);
	}
	
	public long toNode(int pos, int char_pos, int tag_id){
		int[] arr = new int[]{pos, char_pos, tag_id, NodeType.Node.ordinal()};
		return NetworkIDMapper.toHybridNodeID(arr);
	}
	
	public long toNode_root(int size){
		int[] arr = new int[]{size - 1, labels.size(), labels.size(), NodeType.Root.ordinal()};
		return NetworkIDMapper.toHybridNodeID(arr);
	}

	public boolean checkConstraint(int i, int i_j, String prevLabel, int m, int m_n, String nextLabel, boolean isStart, boolean isRoot) {
		if (isStart) {
			if (nextLabel.startsWith("I-") || nextLabel.startsWith("E-")) return false;
			if (nextLabel.equals("O")) return true;
			String nextLabelSuffix = nextLabel.substring(nextLabel.length()-1, nextLabel.length());
			if (nextLabelSuffix.startsWith("I") || nextLabelSuffix.startsWith("E")) return false;
			else return true;
		}
		if (isRoot) {
			if (prevLabel.startsWith("B-") || prevLabel.startsWith("I-")) return false;
			if (prevLabel.equals("O")) return true;
			String prevLabelSuffix = prevLabel.substring(prevLabel.length()-1, prevLabel.length());
			if (prevLabelSuffix.startsWith("I") || prevLabelSuffix.startsWith("B")) return false;
			else return true;
		}
		if (isStart && isRoot) throw new RuntimeException("root and start at the same time?");
		if (i == m) {
			String prevLabelPrefix = prevLabel.length() > 2 ? prevLabel.substring(0, prevLabel.length() - 2) : prevLabel;
			String nextLabelPrefix = nextLabel.length() > 2 ? nextLabel.substring(0, nextLabel.length() - 2) : nextLabel;
			if(!prevLabelPrefix.equals(nextLabelPrefix)) {
				return false;
			} else {
				if (prevLabelPrefix.equals("O")) return true; //same position
				String prevLabelSuffix = prevLabel.substring(prevLabel.length()-1, prevLabel.length());
				String nextLabelSuffix = nextLabel.substring(nextLabel.length()-1, nextLabel.length());
				if (prevLabelSuffix.equals("B") || prevLabelSuffix.equals("I")) {
					return nextLabelSuffix.equals("I") || nextLabelSuffix.equals("E");
				} else if(prevLabelSuffix.equals("S") || prevLabelSuffix.equals("E")) {
					return nextLabelSuffix.equals("S") || nextLabelSuffix.equals("B");
				} else {
					throw new RuntimeException("unknwo ?");
				}
			}
		} else {
			//different word.
			if (m_n != 0) throw new RuntimeException("the next char pos is not 0? but two words?");
			String prevLabelSuffix = prevLabel.substring(prevLabel.length()-1, prevLabel.length());
			String nextLabelSuffix = nextLabel.substring(nextLabel.length()-1, nextLabel.length());
			if (prevLabelSuffix.equals("I") || prevLabelSuffix.equals("B") ) return false;
			if (nextLabelSuffix.equals("I") || nextLabelSuffix.equals("E") ) return false;
			String prevLabelPrefix = prevLabel.length() > 2 ? prevLabel.substring(0, prevLabel.length() - 2) : prevLabel;
			String nextLabelPrefix = nextLabel.length() > 2 ? nextLabel.substring(0, nextLabel.length() - 2) : nextLabel;
			if (prevLabelPrefix.equals("O")) {
				return nextLabelPrefix.startsWith("B-") || nextLabelPrefix.startsWith("S-") || nextLabelPrefix.equals("O");
			}
			if (nextLabelPrefix.equals("O")) {
				return prevLabelPrefix.startsWith("E-") || prevLabelPrefix.startsWith("S-") || prevLabelPrefix.equals("O");
			}
			if (prevLabelPrefix.substring(2).equals(nextLabelPrefix.substring(2))) {
				if (prevLabelPrefix.startsWith("B-") || prevLabelPrefix.startsWith("I-")) {
					return nextLabelPrefix.startsWith("I-") || nextLabelPrefix.startsWith("E-");
				} else if (prevLabelPrefix.startsWith("E-") || prevLabelPrefix.startsWith("S-")) {
					return nextLabelPrefix.startsWith("S-") || nextLabelPrefix.startsWith("B-");
 				} else throw new RuntimeException("unknwo ?222");
			} else {
				if (prevLabelPrefix.startsWith("B-") || prevLabelPrefix.startsWith("I-")) return false;
				else if (prevLabelPrefix.startsWith("E-") || prevLabelPrefix.startsWith("S-")) {
					return nextLabelPrefix.startsWith("S-") || nextLabelPrefix.startsWith("B-");
 				} else throw new RuntimeException("unknwo ?33");
			}
		}
	}
	
	public BaseNetwork compileLabeled(int networkId, Instance instance, LocalNetworkParam param){
		EInst inst = (EInst)instance;
		NetworkBuilder<BaseNetwork> lcrfNetwork = NetworkBuilder.builder();
		long leaf = toNode_leaf();
		long[] children = new long[]{leaf};
		lcrfNetwork.addNode(leaf);
		Sentence sent = inst.getInput();
		for(int i = 0; i < inst.size(); i++){
			String word = sent.get(i).getForm();
			for (int j = 0; j < word.length(); j++) {
				String out_label = inst.getOutput().get(i);
				long[] currentNodes = new long[out_label.equals("O") ? 1: CConfig.Char_suffix.length];
				if (out_label.equals("O")) {
					long node = toNode(i, j, this.labelIndex.get(out_label));
					currentNodes[0] = -1;
					for (long child: children) {
						if (child == -1) continue;
						int[] arr = NetworkIDMapper.toHybridNodeArray(child);
						if (this.checkConstraint(arr[0], arr[1], this.labels.get(arr[2]), i, j, out_label, child == leaf, false) && lcrfNetwork.contains(child)) {
							lcrfNetwork.addNode(node);
							lcrfNetwork.addEdge(node, new long[]{child});
							currentNodes[0] = node;
						}
					}
				} else {
					for (int t = 0; t < CConfig.Char_suffix.length; t++) {
						long node = toNode(i, j, this.labelIndex.get(out_label + CConfig.Char_suffix[t]));
						currentNodes[t] = -1;
						for (long child: children) {
							if (child == -1) continue;
							int[] arr = NetworkIDMapper.toHybridNodeArray(child);
							if (this.checkConstraint(arr[0], arr[1], this.labels.get(arr[2]), i, j, inst.getOutput().get(i) + CConfig.Char_suffix[t],
									child == leaf, false) && lcrfNetwork.contains(child)) {
								lcrfNetwork.addNode(node);
								lcrfNetwork.addEdge(node, new long[]{child});
								currentNodes[t] = node;
							}
						}
					}
				}
				
				children = currentNodes;
			}
		}
		long root = toNode_root(inst.size());
		lcrfNetwork.addNode(root);
		for (long child : children) {
			if (child == -1) continue;
			int[] arr = NetworkIDMapper.toHybridNodeArray(child);
			if (this.checkConstraint(arr[0], arr[1], this.labels.get(arr[2]), inst.size(), inst.size(), "ROOT",
					child == leaf, true)) {
				lcrfNetwork.addEdge(root, new long[]{child});
			}
		}
		BaseNetwork network = lcrfNetwork.build(networkId, inst, param, this);
		if(DEBUG){
			BaseNetwork unlabeled = this.compileUnlabeled(networkId, instance, param);
			if (!unlabeled.contains(network)) {
				System.err.println(instance.getInput().toString());
				System.err.println("not contains");
			}
		}
		return network;
	}
	
	public BaseNetwork compileUnlabeled(int networkId, Instance instance, LocalNetworkParam param){
		EInst inst = (EInst)instance;
		NetworkBuilder<BaseNetwork> lcrfNetwork = NetworkBuilder.builder();
		long leaf = toNode_leaf();
		long[] children = new long[]{leaf};
		lcrfNetwork.addNode(leaf);
		Sentence sent = inst.getInput();
		for(int i = 0; i < inst.size(); i++){
			String word = sent.get(i).getForm();
			for (int j = 0; j < word.length(); j++) {
				long[] currentNodes = new long[this.labels.size()];
				for (int t = 0; t < this.labels.size(); t++) {
					long node = toNode(i, j, t);
					currentNodes[t] = -1;
					for (long child: children) {
						if (child == -1) continue;
						int[] arr = NetworkIDMapper.toHybridNodeArray(child);
						if (this.checkConstraint(arr[0], arr[1], this.labels.get(arr[2]), i, j, this.labels.get(t), child == leaf, false)  && lcrfNetwork.contains(child)) {
							lcrfNetwork.addNode(node);
							lcrfNetwork.addEdge(node, new long[]{child});
							currentNodes[t] = node;
						}
						
					}
				}
				children = currentNodes;
			}
		}
		long root = toNode_root(inst.size());
		lcrfNetwork.addNode(root);
		for (long child : children) {
			if (child == -1) continue;
			int[] arr = NetworkIDMapper.toHybridNodeArray(child);
			if (this.checkConstraint(arr[0], arr[1], this.labels.get(arr[2]), inst.size(), inst.size(), "ROOT",
					child == leaf, true)) {
				lcrfNetwork.addEdge(root, new long[]{child});
			}
		}
		BaseNetwork network = lcrfNetwork.build(networkId, inst, param, this);
		return network;
	}
	
	@Override
	public EInst decompile(Network network) {
		BaseNetwork lcrfNetwork = (BaseNetwork)network;
		EInst lcrfInstance = (EInst)lcrfNetwork.getInstance();
		EInst result = lcrfInstance.duplicate();
		ArrayList<String> prediction = new ArrayList<String>();
		long root = toNode_root(lcrfInstance.size());
		int rootIdx = Arrays.binarySearch(lcrfNetwork.getAllNodes(),root);
		//System.err.println(rootIdx+" final score:"+network.getMax(rootIdx));
		String[] pred_arr = new String[result.size()];
		List<List<String>> chars = new ArrayList<>();
		for (int i = 0; i < pred_arr.length;i++) chars.add(new ArrayList<>());
		while (rootIdx != 0) {
			int child_k = lcrfNetwork.getMaxPath(rootIdx)[0];
			long child = lcrfNetwork.getNode(child_k);
			int[] arr = NetworkIDMapper.toHybridNodeArray(child);
			int word_pos = arr[0];
//			int char_pos = arr[1];
			int tagID = arr[2];
			NodeType type = NodeType.values()[arr[3]];
			if (type == NodeType.Leaf) {
				break;
			}
			String resEntity = this.labels.get(tagID);
			if(resEntity.startsWith(EConf.S_)) resEntity = EConf.B_+resEntity.substring(2);
			if(resEntity.startsWith(EConf.E_)) resEntity = EConf.I_+resEntity.substring(2);
			chars.get(word_pos).add(0, resEntity);
			if(!resEntity.equals(EConf.O)) resEntity = resEntity.substring(0, resEntity.length()-2);
			pred_arr[word_pos] = resEntity;
			rootIdx = child_k;
		}
		for(String entity: pred_arr){
			prediction.add(entity);
		}
		result.setPrediction(prediction);
		result.setDetails(chars);
		return result;
	}
	
	public double costAt(Network network, int parent_k, int[] child_k){
		return 0;
	}
	
}
