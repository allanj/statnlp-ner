package org.statnlp.example.linear_ne;

import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.ArrayList;

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
		word, transition, bigram, trigram};
	
	private WordEmbedding emb;
		
	public ECRFFeatureManager(GlobalNetworkParam param_g, WordEmbedding emb) {
		super(param_g);
		this.emb = emb;
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
		
		String word = inst.getInput().get(pos).getForm();
		fa = this.createFeatureArray(network, featureList);
		FeatureArray curr = fa;
		if(NetworkConfig.USE_NEURAL_FEATURES){
			Object input = null;
			String sentenceInput = sent.toString();
			input = new SimpleImmutableEntry<String, Integer>(sentenceInput, pos);
			this.addNeural(network, 0, parent_k, children_k_index, input, eId);
		} else {
			int embDim = this.emb.getDimension();
			//current word
//			int[] fs1 = new int[embDim];
//			for (int i = 0; i < embDim; i++) {
//				fs1[i] = this._param_g.toFeature(network,FeaType.word.name() + "-"+ i, entity, "");
//			}
//			curr = curr.addNext(this.createFeatureArray(network, fs1, this.emb.getEmbedding(word)));
			
			int[] fs1 = new int[1];
			fs1[0] = this._param_g.toFeature(network,FeaType.word.name(), entity, word);
			curr = curr.addNext(this.createFeatureArray(network, fs1));
			
//			for (int p = pos - 1; p <= pos; p++) {
//				String prevWord = p >= 0 ? sent.get(p).getForm() : "<start>";
//				String currWord = (p + 1) >= sent.length() ? "<end>" : sent.get(p + 1).getForm();
//				double[] pe = this.emb.getEmbedding(prevWord);
//				double[] ce = this.emb.getEmbedding(currWord);
//				int indicator = pos - p;
//				int[] fsbg = new int[embDim * embDim];
//				double[] fsbgval = new double[embDim * embDim];
//				int k = 0;
//				for (int i = 0; i < embDim; i++) {
//					for (int j = 0; j < embDim; j++) {
//						fsbg[k] = this._param_g.toFeature(network,FeaType.bigram.name() + "-"+indicator+":"+ i+":" + j, entity, "");
//						fsbgval[k] = pe[i] * ce[j];
//						k++;
//					}
//				}
//				curr = curr.addNext(this.createFeatureArray(network, fsbg, fsbgval));
//			}
			
//			String prevWord = pos -1 >= 0 ? sent.get(pos -1).getForm() : "<start>";
//			String currWord = sent.get(pos).getForm();
//			String nextWord = (pos + 1) >= sent.length() ? "<end>" : sent.get(pos+ 1).getForm();
//			double[] pe = this.emb.getEmbedding(prevWord);
//			double[] ce = this.emb.getEmbedding(currWord);
//			double[] ne = this.emb.getEmbedding(nextWord);
//			int[] fstg = new int[embDim * embDim * embDim];
//			double[] fstgval = new double[embDim * embDim * embDim];
//			int k = 0;
//			for (int i = 0; i < embDim; i++) {
//				for (int j = 0; j < embDim; j++) {
//					for (int m = 0; m < embDim; m++) {
//						fstg[k] = this._param_g.toFeature(network,FeaType.trigram.name() +":"+ i+":" + j+":"+k, entity, "");
//						fstgval[k] = pe[i] * ce[j] * ne[m];
//						k++;
//					}
//				}
//			}
//			curr = curr.addNext(this.createFeatureArray(network, fstg, fstgval));
		}
		
		return fa;
	}
}
