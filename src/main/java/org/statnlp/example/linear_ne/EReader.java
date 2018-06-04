package org.statnlp.example.linear_ne;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.statnlp.commons.io.RAWF;
import org.statnlp.commons.types.Sentence;
import org.statnlp.commons.types.WordToken;

public class EReader {

	public List<String> labels;
	
	public EReader(List<String> labels) {
		this.labels = labels;
	}
	
	/**
	 * 
	 * @param path
	 * @param setLabel
	 * @param number
	 * @param encoding: IOB, IOBES, NONE (by default it's iob encoding)
	 * @return
	 * @throws IOException
	 */
	public EInst[] readData(String path, boolean isTraining, int number) throws IOException{
		BufferedReader br = RAWF.reader(path);
		String line = null;
		List<EInst> insts = new ArrayList<EInst>();
		int index =1;
		List<WordToken> words = new ArrayList<WordToken>();
		ArrayList<String> output = new ArrayList<String>();
		while((line = br.readLine())!=null){
			if(line.equals("")){
				WordToken[] wordsArr = new WordToken[words.size()];
				words.toArray(wordsArr);
				Sentence sent = new Sentence(wordsArr);
				//postprocess 
				for (int i = 0; i < output.size(); i++) {
					String currEnt = output.get(i);
					if (i == output.size() - 1) {
						if (currEnt.startsWith(EConf.B_)) output.set(i, EConf.S_ + currEnt.substring(2));
						else if (currEnt.startsWith(EConf.I_)) output.set(i, EConf.E_ + currEnt.substring(2));
					} else {
						String nextEnt = output.get(i + 1);
						if (currEnt.startsWith(EConf.B_)) {
							if (nextEnt.equals(EConf.O) || nextEnt.startsWith(EConf.B_)) output.set(i, EConf.S_ + currEnt.substring(2));
						} else if (currEnt.startsWith(EConf.I_)) {
							if (nextEnt.equals(EConf.O) || nextEnt.startsWith(EConf.B_)) output.set(i, EConf.E_ + currEnt.substring(2));
						}
					}
					if (!this.labels.contains(output.get(i))) {
						this.labels.add(output.get(i));
					}
				}
				EInst inst = new EInst(index++, 1.0, sent, output);
				if(isTraining) inst.setLabeled(); else inst.setUnlabeled();
				insts.add(inst);
				words = new ArrayList<WordToken>();
				output = new ArrayList<String>();
				if(number!=-1 && insts.size()==number) break;
				continue;
			} else {
				String[] values = line.split(" ");
				String entity = values[2];
				output.add(entity);
				words.add(new WordToken(values[0],values[1], -1));
			}
		}
		br.close();
		System.err.println("[Info] total:"+ insts.size()+" Instance. ");
		return insts.toArray(new EInst[insts.size()]);
	}
	
	/**
	 * Replace singelton with UNK.
	 * @param instances
	 */
	public void preprocess(EInst[] instances, boolean lowercase, boolean isTraining) {
		Map<String, Integer> wordCount = new HashMap<>();
		for(EInst inst : instances) {
			Sentence sent = inst.getInput();
			for(int p = 0; p < sent.length(); p++) {
				String word = sent.get(p).getForm();
				if (lowercase) {
					sent.get(p).setForm(word.toLowerCase());
					word = sent.get(p).getForm();
				}
				word = word.replaceAll("\\d", "0");
				sent.get(p).setForm(word);
				if (wordCount.containsKey(word)) {
					int num =  wordCount.get(word);
					wordCount.put(word, num + 1);
				} else {
					wordCount.put(word, 1);
				}
			}
		}
	}

}
