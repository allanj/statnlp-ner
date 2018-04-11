package org.statnlp.example.linear_ne;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import org.statnlp.commons.io.RAWF;
import org.statnlp.commons.ml.opt.GradientDescentOptimizer.BestParamCriteria;
import org.statnlp.commons.ml.opt.OptimizerFactory;
import org.statnlp.commons.types.Instance;
import org.statnlp.hypergraph.DiscriminativeNetworkModel;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.NetworkConfig;
import org.statnlp.hypergraph.NetworkModel;
import org.statnlp.hypergraph.decoding.Metric;
import org.statnlp.hypergraph.neural.GlobalNeuralNetworkParam;
import org.statnlp.hypergraph.neural.NeuralNetworkCore;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;

public class EMain {
	
	public static boolean DEBUG = false;

	public static int trainNumber = -100;
	public static int devNumber = -100;
	public static int testNumber = -100;
	public static int numIteration = 100;
	public static int numThreads = 5;
	public static double l2 = 0;
	
	public static String trainFile = "data/conll2003/train.txt";
	public static String devFile = "data/conll2003/test.txt";
	public static String testFile = "data/conll2003/test.txt";
	public static String nerOut = "data/conll2003/output/ner_out.txt";
	public static String tmpOut = "data/conll2003/output/tmp_out.txt";
	public static boolean saveModel = false;
	public static boolean readModel = false;
	public static String modelFile = "models/linearNE.m";
	public static String nnModelFile = "models/lstm.m";
	public static int gpuId = -1;
	public static String nnOptimizer = "sgdclip";
	public static String embedding = "glove";
	public static int batchSize = 10;
	public static OptimizerFactory optimizer = OptimizerFactory.getLBFGSFactory();
	public static boolean evalOnDev = true;
	public static int evalFreq = 1000;
	public static boolean lowercase = true;
	public static boolean fixEmbedding = false;
	public static double dropout = 0.0;
	
	public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException{

		processArgs(args);
		System.err.println("[Info] trainingFile: "+trainFile);
		System.err.println("[Info] testFile: "+testFile);
		System.err.println("[Info] nerOut: "+nerOut);
		
		EInst[] trainInstances = null;
		EInst[] devInstances = null;
		EInst[] testInstances = null;
		List<String> labels = new ArrayList<>();
		
		EReader reader = new EReader(labels);
		trainInstances = reader.readData(trainFile, true, trainNumber);
		System.out.println("[Info] labels:" + labels.toString());
		devInstances = reader.readData(devFile, false, devNumber);
		
		NetworkConfig.CACHE_FEATURES_DURING_TRAINING = true;
		NetworkConfig.L2_REGULARIZATION_CONSTANT = l2;
		NetworkConfig.NUM_THREADS = numThreads;
		NetworkConfig.BATCH_SIZE = batchSize; //need to enable batch training first
		NetworkConfig.RANDOM_BATCH = false;
		NetworkConfig.PRINT_BATCH_OBJECTIVE = false;
		NetworkConfig.FEATURE_TOUCH_TEST = true;
		
		//In order to compare with neural architecture for named entity recognition
		if (DEBUG) {
			NetworkConfig.RANDOM_INIT_WEIGHT = false;
			NetworkConfig.FEATURE_INIT_WEIGHT = 0.1;
		}
		
		NetworkModel model = null;
		if (!readModel) {
			List<NeuralNetworkCore> nets = new ArrayList<NeuralNetworkCore>();
			if(NetworkConfig.USE_NEURAL_FEATURES){
				reader.preprocess(trainInstances, lowercase);
				nets.add(new LampleBiLSTM("SimpleBiLSTM", labels.size(), gpuId, embedding,
						fixEmbedding, dropout)
						.setModelFile(nnModelFile));
			} 
			GlobalNetworkParam gnp = new GlobalNetworkParam(optimizer, new GlobalNeuralNetworkParam(nets));
			ECRFFeatureManager fa = new ECRFFeatureManager(gnp);
			ECRFNetworkCompiler compiler = new ECRFNetworkCompiler(labels);
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
	}
	
	public static void processArgs(String[] args){
		ArgumentParser parser = ArgumentParsers.newArgumentParser("")
				.defaultHelp(true).description("Latent Linear-chain CRGs");
		parser.addArgument("-t", "--thread").type(Integer.class).setDefault(numThreads).help("number of threads");
		parser.addArgument("--train_num").type(Integer.class).setDefault(trainNumber).help("number of training data");
		parser.addArgument("--dev_num").type(Integer.class).setDefault(devNumber).help("number of validation data");
		parser.addArgument("--test_num").type(Integer.class).setDefault(testNumber).help("number of test data");
		parser.addArgument("-it", "--iter").type(Integer.class).setDefault(numIteration).help("number of iterations");
		parser.addArgument("-w", "--windows").type(Boolean.class).setDefault(ECRFEval.windows).help("windows system for running eval script");
		parser.addArgument("-b", "--batch").nargs("*").setDefault(new Object[] {NetworkConfig.USE_BATCH_TRAINING, batchSize}).help("batch training configuration");
		parser.addArgument("-lc", "--lowercase").type(Boolean.class).setDefault(lowercase).help("use lowercase in lstm or not");
		parser.addArgument("-optim", "--optimizer").type(String.class).choices("lbfgs", "sgdclip", "adam").setDefault("lbfgs").help("optimizer");
		parser.addArgument("-emb", "--embedding").type(String.class).choices("glove", "google", "random", "turian").setDefault(embedding).help("embedding to use");
		parser.addArgument("-gi", "--gpuid").type(Integer.class).setDefault(gpuId).help("gpuid");
		parser.addArgument("-l2", "--l2val").type(Double.class).setDefault(l2).help("L2 regularization term");
		parser.addArgument("-ed", "--evalDev").nargs("*").setDefault(new Object[] {evalOnDev, evalFreq}).help("evaluate on dev set");
		parser.addArgument("-lstm", "--useLSTM").type(Boolean.class).setDefault(NetworkConfig.USE_NEURAL_FEATURES).help("use lstm features");
		parser.addArgument("--saveModel", "-sm").type(Boolean.class).setDefault(saveModel).help("save model");
		parser.addArgument("--readModel", "-rm").type(Boolean.class).setDefault(readModel).help("read model");
		parser.addArgument("-fe", "--fixEmbedding").type(Boolean.class).setDefault(fixEmbedding).help("fix embedding");
		parser.addArgument("-do", "--dropout").type(Double.class).setDefault(dropout).help("dropout rate for the lstm");
		parser.addArgument("-os", "--system").type(String.class).setDefault(NetworkConfig.OS).help("system for lua");
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
        numIteration = ns.getInt("iter");
        ECRFEval.windows = ns.getBoolean("windows");
        List<Object> batchOps = ns.getList("batch");
        NetworkConfig.USE_BATCH_TRAINING = batchOps.get(0) instanceof Boolean ? (boolean)batchOps.get(0)
        		: Boolean.valueOf((String)batchOps.get(0));
        batchSize = batchOps.get(1) instanceof Integer ? (int)batchOps.get(1) 
        		:Integer.valueOf((String)batchOps.get(1));
        lowercase = ns.getBoolean("lowercase");
        String optim = ns.getString("optimizer");
        switch (optim) {
        	case "lbfgs": optimizer = OptimizerFactory.getLBFGSFactory(); break;
        	case "sgdclip": optimizer = OptimizerFactory.getGradientDescentFactoryUsingGradientClipping(BestParamCriteria.BEST_ON_DEV, 0.05, 5); break;
        	case "adam" : optimizer = OptimizerFactory.getGradientDescentFactoryUsingAdaM(BestParamCriteria.BEST_ON_DEV); break;
        	default: optimizer = OptimizerFactory.getLBFGSFactory(); break;
        }
        embedding = ns.getString("embedding");
        gpuId = ns.getInt("gpuid");
        l2 = ns.getDouble("l2val");
        List<Object> evalDevOps = ns.getList("evalDev");
        evalOnDev =evalDevOps.get(0) instanceof Boolean ? (boolean)evalDevOps.get(0): Boolean.valueOf((String)evalDevOps.get(0));
        evalFreq = evalDevOps.get(1) instanceof Integer ? (int)evalDevOps.get(1) : Integer.valueOf((String)evalDevOps.get(1));
        NetworkConfig.USE_NEURAL_FEATURES = ns.getBoolean("useLSTM");
        saveModel = ns.getBoolean("saveModel");
        readModel = ns.getBoolean("readModel");
        fixEmbedding = ns.getBoolean("fixEmbedding");
        dropout = ns.getDouble("dropout");
        NetworkConfig.OS = ns.getString("system");
        for (String key : ns.getAttrs().keySet()) {
        	System.err.println(key + "=" + ns.get(key));
        }
	}
}
