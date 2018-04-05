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
import org.statnlp.hypergraph.NetworkConfig.ModelType;
import org.statnlp.hypergraph.NetworkModel;
import org.statnlp.hypergraph.decoding.Metric;
import org.statnlp.hypergraph.neural.GlobalNeuralNetworkParam;
import org.statnlp.hypergraph.neural.NeuralNetworkCore;

public class EMain {
	
	public static boolean DEBUG = false;

	public static int trainNumber = -100;
	public static int devNumber = -100;
	public static int testNumber = -100;
	public static int numIteration = 100;
	public static int numThreads = 5;
	public static String MODEL = "ssvm";
	public static double adagrad_learningRate = 0.1;
	public static double l2 = 0.01;
	
	public static String trainFile = "data/conll2003/train.txt";
	public static String devFile = "data/conll2003/dev.txt";
	public static String testFile = "data/conll2003/test.txt";
	public static String nerOut = "data/conll2003/output/ner_out.txt";
	public static String tmpOut = "data/conll2003/output/tmp_out.txt";
	public static boolean saveModel = false;
	public static boolean readModel = false;
	public static String modelFile = "models/linearNE.m";
	public static String nnModelFile = "models/lstm.m";
	public static String neuralType = "lstm";
	public static boolean iobes = false;
	public static int gpuId = -1;
	public static String nnOptimizer = "lbfgs";
	public static String embedding = "random";
	public static int batchSize = 10;
	public static OptimizerFactory optimizer = OptimizerFactory.getLBFGSFactory();
	public static boolean evalOnDev = false;
	public static int evalFreq = 1000;
	public static boolean lowercase = false;
	
	public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException{

		processArgs(args);
		System.err.println("[Info] trainingFile: "+trainFile);
		System.err.println("[Info] testFile: "+testFile);
		System.err.println("[Info] nerOut: "+nerOut);
		System.err.println("[Info] use IOBES constraint to build network: "+iobes);
		
		EInst[] trainInstances = null;
		EInst[] devInstances = null;
		EInst[] testInstances = null;
		List<String> labels = new ArrayList<>();
		
		EReader reader = new EReader(labels);
		trainInstances = reader.readData(trainFile, true, trainNumber);
		devInstances = reader.readData(devFile, false, devNumber);
		
		NetworkConfig.CACHE_FEATURES_DURING_TRAINING = true;
		NetworkConfig.L2_REGULARIZATION_CONSTANT = l2;
		NetworkConfig.NUM_THREADS = numThreads;
		NetworkConfig.PARALLEL_FEATURE_EXTRACTION = true;
		NetworkConfig.BATCH_SIZE = batchSize; //need to enable batch training first
		NetworkConfig.RANDOM_BATCH = false;
		NetworkConfig.PRINT_BATCH_OBJECTIVE = false;
		
		//In order to compare with neural architecture for named entity recognition
		if (DEBUG) {
			NetworkConfig.RANDOM_INIT_WEIGHT = false;
			NetworkConfig.FEATURE_INIT_WEIGHT = 0.1;
		}
		
		NetworkModel model = null;
		if (!readModel) {
			List<NeuralNetworkCore> nets = new ArrayList<NeuralNetworkCore>();
			if(NetworkConfig.USE_NEURAL_FEATURES){
				int hiddenSize = 100;
				String optimizer = nnOptimizer;
				boolean bidirection = true;
				nets.add(new LampleBiLSTM("SimpleBiLSTM", hiddenSize, bidirection, optimizer, 0.05, 5, labels.size(), gpuId, embedding, 5)
						.setModelFile(nnModelFile));
			} 
			GlobalNetworkParam gnp = new GlobalNetworkParam(optimizer, new GlobalNeuralNetworkParam(nets));
			ECRFFeatureManager fa = new ECRFFeatureManager(gnp, lowercase);
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
		Instance[] predictions = model.decode(testInstances);
		ECRFEval.evalNER(predictions, nerOut);
	}
	
	public static void processArgs(String[] args){
		if(args.length > 0 && ( args[0].equals("-h") || args[0].equals("help") || args[0].equals("-help")) ){
			System.err.println("Linear-Chain CRF Version: Joint DEPENDENCY PARSING and Entity Recognition TASK: ");
			System.err.println("\t usage: java -jar dpe.jar -trainNum -1 -testNum -1 -thread 5 -iter 100 -pipe true");
			System.err.println("\t put numTrainInsts/numTestInsts = -1 if you want to use all the training/testing instances");
			System.exit(0);
		}else{
			for(int i=0;i<args.length;i=i+2){
				switch(args[i]){
					case "-trainNum": trainNumber = Integer.valueOf(args[i+1]); break;   //default: all 
					case "-testNum": testNumber = Integer.valueOf(args[i+1]); break;    //default:all
					case "-devNum": devNumber = Integer.valueOf(args[i+1]); break;    //default:all
					case "-iter": numIteration = Integer.valueOf(args[i+1]); break;   //default:100;
					case "-thread": numThreads = Integer.valueOf(args[i+1]); break;   //default:5
					case "-testFile": testFile = args[i+1]; break;        
					case "-windows":ECRFEval.windows = true; break;            //default: false (is using windows system to run the evaluation script)
					case "-batch": NetworkConfig.USE_BATCH_TRAINING = true;
									batchSize = Integer.valueOf(args[i+1]); break;
					case "-model": NetworkConfig.MODEL_TYPE = args[i+1].equals("crf")? ModelType.CRF:ModelType.SSVM;   break;
					case "-neural": if(args[i+1].equals("mlp") || args[i+1].equals("lstm")|| args[i+1].equals("continuous")
							|| args[i+1].equals("embedding_layer")){ 
											NetworkConfig.USE_NEURAL_FEATURES = true;
											neuralType = args[i+1]; //by default optim_neural is false.
											NetworkConfig.IS_INDEXED_NEURAL_FEATURES = false; //only used when using the senna embedding.
											NetworkConfig.REGULARIZE_NEURAL_FEATURES = true;
									}
									break;
					case "-iobes":  iobes = args[i+1].equals("true") ? true : false; break;
					case "-lowercase":  lowercase = args[i+1].equals("true") ? true : false; break;
					case "-initNNweight": 
						NetworkConfig.INIT_FV_WEIGHTS = args[i+1].equals("true") ? true : false; //optimize the neural features or not
						break;
					case "-optimNeural": 
						NetworkConfig.OPTIMIZE_NEURAL = args[i+1].equals("true") ? true : false; //optimize the neural features or not
						if (!NetworkConfig.OPTIMIZE_NEURAL) {
							nnOptimizer = args[i+2];
							i++;
						}break;
					case "-optimizer":
						 if(args[i+1].equals("sgd")) {
							 System.out.println("[Info] Using SGD with gradient clipping, take best parameter on development set.");
							 optimizer = OptimizerFactory.getGradientDescentFactoryUsingGradientClipping(BestParamCriteria.BEST_ON_DEV, 0.05, 5);
						 }
						break;
					case "-emb" : embedding = args[i+1]; break;
					case "-gpuid": gpuId = Integer.valueOf(args[i+1]); break;
					case "-reg": l2 = Double.valueOf(args[i+1]);  break;
					case "-lr": adagrad_learningRate = Double.valueOf(args[i+1]); break;
					case "-backend": NetworkConfig.NEURAL_BACKEND = args[i+1]; break;
					case "-os": NetworkConfig.OS = args[i+1]; break; // for Lua native lib, "osx" or "linux"
					case "-evalDev": evalOnDev = args[i+1].equals("true") ? true : false; 
						if (evalOnDev) {
							evalFreq = Integer.valueOf(args[i+2]);
							i++;
						}
						break;
					default: System.err.println("Invalid arguments "+args[i]+", please check usage."); System.exit(0);
				}
			}
			System.err.println("[Info] trainNum: "+trainNumber);
			System.err.println("[Info] testNum: "+testNumber);
			System.err.println("[Info] numIter: "+numIteration);
			System.err.println("[Info] numThreads: "+numThreads);
			System.err.println("[Info] Regularization Parameter: "+l2);
		}
	}
}
