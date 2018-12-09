package org.statnlp.example.linear_ne;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.statnlp.commons.io.RAWF;
import org.statnlp.commons.types.Sentence;
import org.statnlp.commons.types.WordToken;

public class EReader {

	public List<String> labels;
	public boolean characterModel;
	public String dataset;
	
	public EReader(List<String> labels, boolean characterModel, String dataset) {
		this.labels = labels;
		this.characterModel = characterModel;
		this.dataset = dataset;
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
		int maxLen = -1;
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
					if (!this.characterModel) {
						if (isTraining && !this.labels.contains(output.get(i))) {
							this.labels.add(output.get(i));
						}
					} else {
						if (isTraining) {
							if (output.get(i).equals(EConf.O)  ) {
								if ( !this.labels.contains(output.get(i)))
									this.labels.add(output.get(i));
							} else {
								if (!this.labels.contains(output.get(i)+ "-B") ) this.labels.add(output.get(i) + "-B");
								if (!this.labels.contains(output.get(i)+ "-I") ) this.labels.add(output.get(i) + "-I");
								if (!this.labels.contains(output.get(i)+ "-E") ) this.labels.add(output.get(i) + "-E");
								if (!this.labels.contains(output.get(i)+ "-S") ) this.labels.add(output.get(i) + "-S");
								
							}
						}
					}
					
				}
				maxLen = Math.max(maxLen, sent.length());
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
				String posTag = values[1];
				words.add(new WordToken(values[0],posTag, -1));
			}
		}
		br.close();
		System.err.println("[Info] total:"+ insts.size()+" Instance. ");
		System.err.println("[Info] maximum length:"+ maxLen);
		return insts.toArray(new EInst[insts.size()]);
	}

}
