package org.statnlp.example.linear_ne;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.statnlp.commons.types.Instance;
import org.statnlp.example.base.BaseNetwork;
import org.statnlp.example.base.BaseNetwork.NetworkBuilder;
import org.statnlp.hypergraph.LocalNetworkParam;
import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkCompiler;
import org.statnlp.hypergraph.NetworkIDMapper;

public class ECRFNetworkCompiler extends NetworkCompiler{

	private static final long serialVersionUID = -2388666010977956073L;

	public enum NodeType {Leaf, Node, Root};
	public static final int _size = 150;
	public BaseNetwork genericUnlabeledNetwork;
	//0: should be start tag
	//length -1: should be end tag;
	private List<String> labels;
	private Map<String, Integer> labelIndex;
	private final boolean DEBUG = false;
	
	static {
		NetworkIDMapper.setCapacity(new int[]{_size, 50, 3});
	}
	
	public ECRFNetworkCompiler(List<String> labels){
		this.labels = labels;
		this.labelIndex = new HashMap<>(this.labels.size());
		for (int l = 0; l < this.labels.size(); l++) {
			this.labelIndex.put(this.labels.get(l), l);
		}
		this.compileUnlabeledInstancesGeneric();
	}
	
	public long toNode_leaf(){
		//since 0 is the start_tag index;
		int[] arr = new int[]{0, 0, NodeType.Leaf.ordinal()};
		return NetworkIDMapper.toHybridNodeID(arr);
	}
	
	public long toNode(int pos, int tag_id){
		int[] arr = new int[]{pos, tag_id, NodeType.Node.ordinal()};
		return NetworkIDMapper.toHybridNodeID(arr);
	}
	
	public long toNode_root(int size){
		int[] arr = new int[]{size - 1, labels.size(), NodeType.Root.ordinal()};
		return NetworkIDMapper.toHybridNodeID(arr);
	}

	public BaseNetwork compileLabeled(int networkId, Instance instance, LocalNetworkParam param){
		EInst inst = (EInst)instance;
		NetworkBuilder<BaseNetwork> lcrfNetwork = NetworkBuilder.builder();
		long leaf = toNode_leaf();
		long[] children = new long[]{leaf};
		lcrfNetwork.addNode(leaf);
		for(int i = 0; i < inst.size(); i++){
			long node = toNode(i, this.labelIndex.get( inst.getOutput().get(i) ));
			lcrfNetwork.addNode(node);
			long[] currentNodes = new long[]{node};
			lcrfNetwork.addEdge(node, children);
			children = currentNodes;
		}
		long root = toNode_root(inst.size());
		lcrfNetwork.addNode(root);
		lcrfNetwork.addEdge(root, children);
		BaseNetwork network = lcrfNetwork.build(networkId, inst, param, this);
		if(DEBUG){
			if (!genericUnlabeledNetwork.contains(network))
				System.err.println("not contains");
		}
		return network;
	}
	
	public BaseNetwork compileUnlabeled(int networkId, Instance inst, LocalNetworkParam param){
		long[] allNodes = genericUnlabeledNetwork.getAllNodes();
		long root = toNode_root(inst.size());
		int rootIdx = Arrays.binarySearch(allNodes, root);
		if (rootIdx < 0) {
			System.out.println(allNodes.length);
			System.out.println(inst.getInput());
			System.out.println(inst.size());
		}
		BaseNetwork lcrfNetwork = NetworkBuilder.quickBuild(networkId, inst, allNodes, genericUnlabeledNetwork.getAllChildren(), rootIdx+1, param, this);
		return lcrfNetwork;
	}
	
	public void compileUnlabeledInstancesGeneric(){
		NetworkBuilder<BaseNetwork> lcrfNetwork = NetworkBuilder.builder();
		long leaf = toNode_leaf();
		long[] children = new long[]{leaf};
		lcrfNetwork.addNode(leaf);
		for(int i = 0; i < _size; i++){
			long[] currentNodes = new long[this.labels.size()];
			for(int l = 0; l < this.labels.size(); l++){
				long node = toNode(i,l);
				for(long child: children){
					if(lcrfNetwork.contains(child)){
						lcrfNetwork.addNode(node);
						lcrfNetwork.addEdge(node, new long[]{child});
					}
				}
				currentNodes[l] = node;
			}
			long root = toNode_root(i + 1);
			lcrfNetwork.addNode(root);
			for(long child : currentNodes){
				lcrfNetwork.addEdge(root, new long[]{child});
			}
			children = currentNodes;
		}
		BaseNetwork network = lcrfNetwork.buildRudimentaryNetwork();
		genericUnlabeledNetwork =  network;
		System.out.println("nodes:" + genericUnlabeledNetwork.getAllNodes().length);
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
		for(int i=0;i<lcrfInstance.size();i++){
			int child_k = lcrfNetwork.getMaxPath(rootIdx)[0];
			long child = lcrfNetwork.getNode(child_k);
			rootIdx = child_k;
			int tagID = NetworkIDMapper.toHybridNodeArray(child)[1];
			String resEntity = this.labels.get(tagID);
			if(resEntity.startsWith(EConf.S_)) resEntity = EConf.B_+resEntity.substring(2);
			if(resEntity.startsWith(EConf.E_)) resEntity = EConf.I_+resEntity.substring(2);
			prediction.add(0, resEntity);
		}
		result.setPrediction(prediction);
		return result;
	}
	
	public double costAt(Network network, int parent_k, int[] child_k){
		return super.costAt(network, parent_k, child_k);
	}
	
}
