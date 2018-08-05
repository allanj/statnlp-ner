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
	
	private transient GloveWordEmbedding emb;
	private boolean useDiscrete;
	private boolean useBigram = true;
	private int wordHalfWindow = 1;
	
	public ECRFFeatureManager(GlobalNetworkParam param_g, GloveWordEmbedding emb, boolean discrete, boolean bigram,
			int wordHalfWindowSize) {
		super(param_g);
		this.emb = emb;
		this.useDiscrete = discrete;
		this.useBigram = bigram;
		this.wordHalfWindow = wordHalfWindowSize;
	}
	
	public void setEmb(GloveWordEmbedding emb) {
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
		
//		String word = inst.getInput().get(pos).getForm();
		fa = this.createFeatureArray(network, featureList);
		FeatureArray curr = fa;
		if(NetworkConfig.USE_NEURAL_FEATURES){
			Object input = null;
			String sentenceInput = sent.toString();
			input = new SimpleImmutableEntry<String, Integer>(sentenceInput, pos);
			this.addNeural(network, 0, parent_k, children_k_index, input, eId);
		} else {
			int embDim = -1;
			if (this.emb != null) {
				embDim = this.emb.getDimension();
				//current word
				for (int w = -wordHalfWindow; w<= wordHalfWindow; w++) {
					String wordNow = ((pos+w >= sent.length()) || (pos+w<0))? "unk" : sent.get(pos + w).getForm();
					int[] fs1 = new int[embDim];
					for (int i = 0; i < embDim; i++) {
						fs1[i] = this._param_g.toFeature(network,FeaType.word.name() + "-" + w + "-"+ i, entity, "");
					}
					curr = curr.addNext(this.createFeatureArray(network, fs1, this.emb.getEmbedding(wordNow)));
					if (this.useDiscrete) {
						int[] fs0 = new int[1];
						fs0[0] = this._param_g.toFeature(network,FeaType.word.name() + "-" + w , entity, wordNow);
						curr = curr.addNext(this.createFeatureArray(network, fs0));
					}
				}
			}
			
			if (this.useBigram) {
				for (int p = pos - wordHalfWindow; p < pos + wordHalfWindow; p++) {
					String prevWord = p >= 0 && p < sent.length() ? sent.get(p).getForm() : "unk";
					String currWord = ((p + 1) >= sent.length() || (p+1<0)) ? "unk" : sent.get(p + 1).getForm();
					String bigram = prevWord.toLowerCase() + " " + currWord.toLowerCase();
					int indicator = pos - p;
//					if (this.emb != null) {
//						int[] fsbg = new int[this.emb.bigramDim];
//						int k = 0;
//						for (int i = 0; i < embDim; i++) {
//							for (int j = 0; j < embDim; j++) {
//								fsbg[k++] = this._param_g.toFeature(network,FeaType.bigram.name() + "-"+indicator+":"+ i+":" + j, entity, "");
//							}
//						}
//						curr = curr.addNext(this.createFeatureArray(network, fsbg, this.emb.getBigramEmbedding(bigram)));
//					}
					if (this.emb != null) {
						int[] fsbg = new int[this.emb.bigramDim];
						int k = 0;
						for (int i = 0; i < this.emb.bigramDim; i++) {
							fsbg[k++] = this._param_g.toFeature(network,FeaType.bigram.name() + "-"+indicator+":"+ i, entity, "");
						}
						curr = curr.addNext(this.createFeatureArray(network, fsbg, this.emb.getBigramEmbedding(bigram)));
					}
					if (this.useDiscrete) {
						int[] disfsbg = new int[1];
						disfsbg[0] = this._param_g.toFeature(network, FeaType.bigram.name() + "-"+indicator, entity, prevWord + " " + currWord);
						curr = curr.addNext(this.createFeatureArray(network, disfsbg));
					}
				}
				
			}
			
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
