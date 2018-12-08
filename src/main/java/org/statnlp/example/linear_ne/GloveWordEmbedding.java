package org.statnlp.example.linear_ne;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;

import org.statnlp.commons.io.RAWF;
import org.statnlp.commons.ml.opt.MathsVector;
import org.statnlp.commons.types.Instance;
import org.statnlp.commons.types.Sentence;

public class GloveWordEmbedding implements WordEmbedding{

	private Map<String, double[]> lookupTable;
	
	protected Map<String, double[]> bigramLookupTable;
	protected Map<String, double[]> trigramLookupTable;
	
	private static final int  dim = 100;
	
	public int bigramDim;
	
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
	
	public void readBigramEmbedding(String file, int dimension) {
		this.bigramLookupTable = new HashMap<>();
		this.bigramDim = dimension;
		BufferedReader br;
		try {
			br = RAWF.reader(file);
			String line = null;
			while((line = br.readLine()) != null) {
				String[] vals = line.split(" ");
				String word = vals[0] + " " + vals[1];
				if (vals.length != 102) throw new RuntimeException("not 102?");
				double[] emb = new double[dimension];
				for (int i = 0; i < emb.length; i++) {
					emb[i] = Double.valueOf(vals[i + 2]);
				}
				if (this.normalized) {
					double norm = MathsVector.norm(emb);
					for (int i = 0; i < emb.length; i++) {
						emb[i] /= norm;
					}
				}
				this.bigramLookupTable.put(word, emb);
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.out.println("[info] finishing bigram embedding..");
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
//		for (Instance in : insts) {
//			EInst inst = (EInst)in;
//			Sentence sent = inst.getInput();
//			for (int i = 0; i <= sent.length(); i++) {
//				String prevWord = i-1 >= 0 ? sent.get(i-1).getForm().toLowerCase(): "unk";
//				String currWord = i < sent.length() ? sent.get(i).getForm().toLowerCase() : "unk";
//				String bigram = prevWord + " " + currWord;
//				if (!this.bigramLookupTable.containsKey(bigram)) {
//					double[] pe = this.getEmbedding(prevWord);
//					double[] ce = this.getEmbedding(currWord);
//					double[] bg = new double[pe.length * ce.length];
//					int k = 0;
//					for (int m = 0; m < pe.length; m++) {
//						for (int n = 0; n < ce.length; n++) {
//							bg[k++] = pe[m] * ce[n];
//						}
//					}
//					if (normalized) {
//						double norm = MathsVector.norm(bg);
//						for (int l = 0; l < bg.length; l++) {
//							bg[l] /= norm;
//						}
//					}
//					this.bigramLookupTable.put(bigram, bg);
//				}
//			}
//		}
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
					double[] bg = new double[dim];
					for (int n = 0; n < dim; n++) {
						bg[n] = pe[n] + ce[n];
					}
					if (normalized) {
						double norm = MathsVector.norm(bg);
						for (int l = 0; l < bg.length; l++) {
							bg[l] /= norm;
						}
					}
					this.bigramLookupTable.put(bigram, bg);
				}
			}
		}
		System.out.println("[Info] Current size of bigram embedding: " + this.bigramLookupTable.size());
	}
	
	public void writeEmbToFile(Map<String, double[]> emb, String file) throws IOException {
		PrintWriter pw = RAWF.writer(file);
		for (String key : emb.keySet()) {
			double[] embedding = emb.get(key);
			pw.write(key);
			for (int i = 0; i < embedding.length; i++) {
				pw.write(" " + embedding[i]);
			}
			pw.println();
		}
		pw.close();
	}

	public void collectTrigramLT(Instance[] insts, boolean normalized) {
		if (this.trigramLookupTable == null) this.trigramLookupTable = new HashMap<>();
//		for (Instance in : insts) {
//			EInst inst = (EInst)in;
//			Sentence sent = inst.getInput();
//			for (int i = 0; i <= sent.length() + 1; i++) {
//				String llw = i-2 >= 0 ? sent.get(i-2).getForm().toLowerCase(): "unk";
//				String prevWord = i-1 >= 0 && i -1 < sent.length() ? sent.get(i-1).getForm().toLowerCase(): "unk";
//				String currWord = i < sent.length() ? sent.get(i).getForm().toLowerCase() : "unk";
//				String trigram = llw + " " + prevWord + " " + currWord;
//				if (!this.trigramLookupTable.containsKey(trigram)) {
//					double[] ppe = this.getEmbedding(llw);
//					double[] pe = this.getEmbedding(prevWord);
//					double[] ce = this.getEmbedding(currWord);
//					double[] tg = new double[ppe.length * pe.length * ce.length];
//					int k = 0;
//					for (int m = 0; m < ppe.length; m++) {
//						for (int n = 0; n < pe.length; n++) {
//							for(int l =0; l < ce.length; l++) {
//								tg[k++] = ppe[m] * pe[n] * ce[l];
//							}
//						}
//					}
//					if (normalized) {
//						double norm = MathsVector.norm(tg);
//						for (int l = 0; l < tg.length; l++) {
//							tg[l] /= norm;
//						}
//					}
//					this.trigramLookupTable.put(trigram, tg);
//				}
//			}
//		}
		for (Instance in : insts) {
			EInst inst = (EInst)in;
			Sentence sent = inst.getInput();
			for (int i = 0; i <= sent.length() + 1; i++) {
				String llw = i-2 >= 0 ? sent.get(i-2).getForm().toLowerCase(): "unk";
				String prevWord = i-1 >= 0 && i -1 < sent.length() ? sent.get(i-1).getForm().toLowerCase(): "unk";
				String currWord = i < sent.length() ? sent.get(i).getForm().toLowerCase() : "unk";
				String trigram = llw + " " + prevWord + " " + currWord;
				if (!this.trigramLookupTable.containsKey(trigram)) {
					double[] ppe = this.getEmbedding(llw);
					double[] pe = this.getEmbedding(prevWord);
					double[] ce = this.getEmbedding(currWord);
					double[] tg = new double[dim];
					for (int m = 0; m < dim; m++) {
						tg[m] = ppe[m] + pe[m] + ce[m];
					}
					if (normalized) {
						double norm = MathsVector.norm(tg);
						for (int l = 0; l < tg.length; l++) {
							tg[l] /= norm;
						}
					}
					this.trigramLookupTable.put(trigram, tg);
				}
			}
		}
		System.out.println("[Info] Current size of trigram embedding: " + this.trigramLookupTable.size());
	}
}
