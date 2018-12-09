package org.statnlp.example.linear_ne;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import org.statnlp.commons.io.RAWF;
import org.statnlp.commons.ml.opt.OptimizerFactory;
import org.statnlp.commons.types.Instance;
import org.statnlp.example.linear_ne.character.CharacterFeatureManager;
import org.statnlp.example.linear_ne.character.CharacterNetworkCompiler;
import org.statnlp.hypergraph.DiscriminativeNetworkModel;
import org.statnlp.hypergraph.FeatureManager;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.NetworkCompiler;
import org.statnlp.hypergraph.NetworkConfig;
import org.statnlp.hypergraph.NetworkModel;
import org.statnlp.hypergraph.decoding.Metric;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;

public class EMain {
	
	public static boolean DEBUG = false;

	public static int trainNumber = 500;
	public static int devNumber = 500;
	public static int testNumber = 500;
	public static int numIteration = 4000;
	public static int numThreads = 8;
	public static double l2 = 0.01;
	
	public static String dataset = "conll2003";
	public static boolean saveModel = false;
	public static boolean readModel = false;
	public static OptimizerFactory optimizer = OptimizerFactory.getLBFGSFactory();
	public static boolean evalOnDev = false;
	public static int evalFreq = 1000;
	public static boolean useCharFeats = false;
	public static enum Model_Type {
		linear,
		latent_character
	};
	public static Model_Type current_model = Model_Type.latent_character;
	
	public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException{

		processArgs(args);
		
		String trainFile = "data/"+dataset+"/train.txt";
		String devFile = "data/"+dataset+"/dev.txt";
		String testFile = "data/"+dataset+"/test.txt";
		String nerOut = "data/"+dataset+"."+current_model.name()+".prefix_"+useCharFeats+".results.txt";
		String nerDetailout = "data/"+dataset+"."+current_model.name()+".prefix_"+useCharFeats+".detail.results.txt";
		String tmpOut = "data/"+dataset+"/"+current_model.name()+".prefix_"+useCharFeats+".tmp_out.txt";
		String modelFile = "models/"+dataset+"."+current_model.name()+".prefix_"+useCharFeats+".m";
		
		System.err.println("[Info] trainingFile: "+trainFile);
		System.err.println("[Info] testFile: "+testFile);
		
		System.err.println("[Info] nerOut: "+nerOut);
		
		EInst[] trainInstances = null;
		EInst[] devInstances = null;
		EInst[] testInstances = null;
		List<String> labels = new ArrayList<>();
		
		
		
		EReader reader = new EReader(labels, current_model == Model_Type.latent_character, dataset);
		trainInstances = reader.readData(trainFile, true, trainNumber);
		System.out.println("[Info] labels:" + labels.toString());
		devInstances = reader.readData(devFile, false, devNumber);
		
		NetworkConfig.L2_REGULARIZATION_CONSTANT = l2;
		NetworkConfig.NUM_THREADS = numThreads;
		
		//In order to compare with neural architecture for named entity recognition
		if (DEBUG) {
			NetworkConfig.RANDOM_INIT_WEIGHT = false;
			NetworkConfig.FEATURE_INIT_WEIGHT = 0.1;
		}
		
		NetworkModel model = null;
		if (!readModel) {
			GlobalNetworkParam gnp = new GlobalNetworkParam(optimizer);
			FeatureManager fa =  current_model == Model_Type.linear ? new ECRFFeatureManager(gnp, useCharFeats) :
				new CharacterFeatureManager(gnp, labels, useCharFeats);
			NetworkCompiler compiler = current_model == Model_Type.linear ? new ECRFNetworkCompiler(labels) :
				new CharacterNetworkCompiler(labels);
			model = DiscriminativeNetworkModel.create(fa, compiler);
			Function<Instance[], Metric> evalFunc = new Function<Instance[], Metric>() {
				@Override
				public Metric apply(Instance[] t) {
					return ECRFEval.evalNER(t, tmpOut);
				}
			};
			if (!evalOnDev) devInstances = null;
			model.train(trainInstances, numIteration, devInstances, evalFunc, evalFreq);
		} else {
			ObjectInputStream ois = RAWF.objectReader(modelFile);
			model = (NetworkModel)ois.readObject();
			ois.close();
		}
		
		if (saveModel) {
			ObjectOutputStream oos =  RAWF.objectWriter(modelFile);
			oos.writeObject(model);
			oos.close();
		}
		testInstances = reader.readData(testFile, false, testNumber);
		Instance[] predictions = model.test(testInstances);
		ECRFEval.evalNER(predictions, nerOut);
		//print details
		if (current_model == Model_Type.latent_character) {
			PrintWriter pw = RAWF.writer(nerDetailout);
			for(Instance prediction : predictions) {
				EInst inst = (EInst)prediction;
				List<List<String>> details = inst.detailedCharacterPrediction;
				for (int i = 0; i < details.size(); i++) {
					pw.print(inst.getInput().get(i).getForm() + " " + inst.prediction.get(i) + " CHARS:" );
					for (int j =0; j < details.get(i).size(); j++) {
						pw.print(" " + details.get(i).get(j));
					}
					pw.println();
				}
				pw.println();
			}
			pw.close();
		}
	}
	
	public static void processArgs(String[] args){
		ArgumentParser parser = ArgumentParsers.newArgumentParser("")
				.defaultHelp(true).description("Latent Linear-chain CRGs");
		parser.addArgument("-t", "--thread").type(Integer.class).setDefault(numThreads).help("number of threads");
		parser.addArgument("--train_num").type(Integer.class).setDefault(trainNumber).help("number of training data");
		parser.addArgument("--dev_num").type(Integer.class).setDefault(devNumber).help("number of validation data");
		parser.addArgument("--test_num").type(Integer.class).setDefault(testNumber).help("number of test data");
		parser.addArgument("--dataset").type(String.class).setDefault(dataset).help("training file");
		parser.addArgument("-it", "--iter").type(Integer.class).setDefault(numIteration).help("number of iterations");
		parser.addArgument("-w", "--windows").type(Boolean.class).setDefault(ECRFEval.windows).help("windows system for running eval script");
		parser.addArgument("-l2", "--l2val").type(Double.class).setDefault(l2).help("L2 regularization term");
		parser.addArgument("-ed", "--evalDev").nargs("*").setDefault(new Object[] {evalOnDev, evalFreq}).help("evaluate on dev set");
		parser.addArgument("-lstm", "--useLSTM").type(Boolean.class).setDefault(NetworkConfig.USE_NEURAL_FEATURES).help("use lstm features");
		parser.addArgument("--saveModel", "-sm").type(Boolean.class).setDefault(saveModel).help("save model");
		parser.addArgument("--readModel", "-rm").type(Boolean.class).setDefault(readModel).help("read model");
		parser.addArgument("-os", "--system").type(String.class).setDefault(NetworkConfig.OS).help("system for lua");
		parser.addArgument("-ucf", "--useCharFeats").type(Boolean.class).setDefault(useCharFeats).help("use prefix/suffix features");
		parser.addArgument("-mt", "--modelType").type(Model_Type.class).setDefault(current_model).help("Model type to use: linear or latent_character");
		Namespace ns = null;
        try {
            ns = parser.parseArgs(args);
        } catch (ArgumentParserException e) {
            parser.handleError(e);
            System.exit(1);
        }
        numThreads = ns.getInt("thread");
        trainNumber = ns.getInt("train_num");
        devNumber = ns.getInt("dev_num");
        testNumber = ns.getInt("test_num");
        dataset = ns.getString("dataset");
        numIteration = ns.getInt("iter");
        ECRFEval.windows = ns.getBoolean("windows");
        l2 = ns.getDouble("l2val");
        List<Object> evalDevOps = ns.getList("evalDev");
        evalOnDev =evalDevOps.get(0) instanceof Boolean ? (boolean)evalDevOps.get(0): Boolean.valueOf((String)evalDevOps.get(0));
        evalFreq = evalDevOps.get(1) instanceof Integer ? (int)evalDevOps.get(1) : Integer.valueOf((String)evalDevOps.get(1));
        saveModel = ns.getBoolean("saveModel");
        readModel = ns.getBoolean("readModel");
        useCharFeats = ns.getBoolean("useCharFeats");
        current_model = (Model_Type)ns.get("modelType");
        for (String key : ns.getAttrs().keySet()) {
        	System.err.println(key + "=" + ns.get(key));
        }
	}
}
