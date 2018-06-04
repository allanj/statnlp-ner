package org.statnlp.example.linear_ne;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.List;

import org.statnlp.commons.io.RAWF;
import org.statnlp.commons.types.Instance;
import org.statnlp.commons.types.Sentence;
import org.statnlp.hypergraph.decoding.Metric;

public class ECRFEval {
	
	
	public static boolean windows = false;
	public static String evalScript = "eval/conlleval.pl";  //remember to make the script runnable
	
	/**
	 * 
	 * @param testInsts
	 * @param nerOut: word, true pos, true entity, pred entity
	 * @throws IOException
	 */
	public static Metric evalNER(Instance[] testInsts, String nerOut){
		PrintWriter pw;
		try {
			pw = RAWF.writer(nerOut);
			for(int index=0;index<testInsts.length;index++){
				EInst eInst = (EInst)testInsts[index];
				List<String> predEntities = eInst.getPrediction();
				List<String> trueEntities = eInst.getOutput();
				Sentence sent = eInst.getInput();
				for(int i=0;i<sent.length();i++){
					String goldEntity = trueEntities.get(i).replace(EConf.E_, EConf.I_);
					goldEntity = goldEntity.replace(EConf.S_, EConf.B_);
					pw.write(sent.get(i).getForm()+" "+sent.get(i).getTag()+" "+goldEntity+" "+predEntities.get(i)+"\n");
				}
				pw.write("\n");
				
			}
			pw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return evalNER(nerOut);
	}
	
	private static Metric evalNER(String outputFile){
		double fscore = 0;
		try{
			System.err.println("perl "+evalScript+" < "+outputFile);
			ProcessBuilder pb = null;
			if(windows){
				pb = new ProcessBuilder("D:/Perl64/bin/perl","eval/conlleval.pl"); 
			}else{
				pb = new ProcessBuilder(evalScript); 
			}
			pb.redirectInput(new File(outputFile));
//			pb.redirectOutput(Redirect.INHERIT);
//			pb.redirectError(Redirect.INHERIT);
			Process process = pb.start();
			BufferedReader br = new BufferedReader(new InputStreamReader(process.getInputStream(), "UTF-8")) ;
			while (!br.ready()) ; // wait until buffered reader is ready.
			while (br.ready()) {
				String line = br.readLine();
				System.out.println(line);
				if (line.startsWith("accuracy")) {
					String[] vals = line.trim().split("\\s+");
					fscore = Double.valueOf(vals[vals.length - 1]);
				}
			}
			br.close();
		}catch(IOException ioe){
			ioe.printStackTrace();
		}
		return new NEMetric(fscore);
	}
	
	public static void writeNERResult(Instance[] predictions, String nerResult, boolean isNERInstance) throws IOException{
		PrintWriter pw = RAWF.writer(nerResult);
		for(int index=0;index<predictions.length;index++){
			Instance inst = predictions[index];
			EInst eInst = (EInst)inst;
			List<String> predEntities = eInst.getPrediction();
			List<String> trueEntities = eInst.getOutput();
			Sentence sent = eInst.getInput();
			for(int i=0;i<sent.length();i++){
				int headIndex = sent.get(i).getHeadIndex()+1;
				pw.write((i+1)+" "+sent.get(i).getForm()+" "+sent.get(i).getTag()+" "+trueEntities.get(i)+" "+predEntities.get(i)+" "+headIndex+"\n");
			}
			pw.write("\n");
		} 
		
		pw.close();
	}
	
}
