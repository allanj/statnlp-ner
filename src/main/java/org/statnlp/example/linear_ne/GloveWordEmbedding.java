package org.statnlp.example.linear_ne;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.statnlp.commons.io.RAWF;
import org.statnlp.commons.ml.opt.MathsVector;
import org.statnlp.commons.types.Instance;
import org.statnlp.commons.types.Sentence;

public class GloveWordEmbedding implements WordEmbedding{

	private Map<String, double[]> lookupTable;
	
	private Map<String, double[]> bigramLookupTable;
	
	private static final int  dim = 100;
	
	private boolean normalized = true;
	
	public GloveWordEmbedding(String file, boolean normalized) {
		this.normalized = normalized;
		System.out.println("[Glove] Loading Glove embeddings....");
		System.out.println("[Glove] Normalize Glove embeddings: " + this.normalized);
		this.readEmbedding(file);
		System.out.println("[Glove] Finish reading Glove embeddings.");
		System.out.println("[Glove] If a word appear in embedding but not in training data, we still use the embedding");
	}
	
	public void readEmbedding(String file) {
		lookupTable = new HashMap<>();
		BufferedReader br;
		try {
			br = RAWF.reader(file);
			String line = null;
			while((line = br.readLine()) != null) {
				String[] vals = line.split(" ");
				String word = vals[0];
				double[] emb = new double[dim];
				for (int i = 0; i < emb.length; i++) {
					emb[i] = Double.valueOf(vals[i + 1]);
				}
				if (this.normalized) {
					double norm = MathsVector.norm(emb);
					for (int i = 0; i < emb.length; i++) {
						emb[i] /= norm;
					}
				}
				this.lookupTable.put(word, emb);
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void normalizeEmbedding() {
		for (String key : this.lookupTable.keySet()) {
			double[] emb = this.lookupTable.get(key);
			double norm = MathsVector.norm(emb);
			for (int i = 0; i < emb.length; i++) {
				emb[i] /= norm;
			}
		}
	}
	
	public double[] getEmbedding(String word) {
		String x = word.toLowerCase();
		return lookupTable.containsKey(x) ?  lookupTable.get(x) : lookupTable.get("unk");
	}
	
	public double[] getBigramEmbedding(String bigram) {
		String x = bigram.toLowerCase();
		return this.bigramLookupTable.get(x);
	}
	
	public void clearEmbeddingMemory() {
		this.lookupTable.clear();
		this.lookupTable = null;
	}

	@Override
	public int getDimension() {
		return dim;
	}

	public void collectBigramLT(Instance[] insts, boolean normalized) {
		if (this.bigramLookupTable == null) this.bigramLookupTable = new HashMap<>();
		for (Instance in : insts) {
			EInst inst = (EInst)in;
			Sentence sent = inst.getInput();
			for (int i = 0; i <= sent.length(); i++) {
				String prevWord = i-1 >= 0 ? sent.get(i-1).getForm().toLowerCase(): "unk";
				String currWord = i < sent.length() ? sent.get(i).getForm().toLowerCase() : "unk";
				String bigram = prevWord + " " + currWord;
				if (!this.bigramLookupTable.containsKey(bigram)) {
					double[] pe = this.getEmbedding(prevWord);
					double[] ce = this.getEmbedding(currWord);
					double[] bg = new double[pe.length * ce.length];
					int k = 0;
					for (int m = 0; m < pe.length; m++) {
						for (int n = 0; n < ce.length; n++) {
							bg[k++] = pe[m] * ce[n];
						}
					}
					if (normalized) {
						double norm = MathsVector.norm(bg);
						for (int l = 0; l < bg.length; l++) {
							bg[i] /= norm;
						}
					}
					this.bigramLookupTable.put(bigram, bg);
				}
			}
		}
		System.out.println("[Info] Current size of bigram embedding: " + this.bigramLookupTable.size());
	}
}
