import java.io.File;
import java.io.FileWriter;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class Prediction {

	private Instances trainDataSource;
	private Instances testDataSource;

	public Prediction() {
		initDataSources();
	}

	void initDataSources() {

		trainDataSource = initDatasource("resources/train.arff");
		
			
			testDataSource = initDatasource("resources/test.arff");


	}

	Instances initDatasource(String file) {

		DataSource source = null;
		try {
			source = new DataSource(file);
		
		} catch (Exception e1) {
			e1.printStackTrace();
		}
		Instances data = null;
		try {
			data = source.getDataSet();
		} catch (Exception e) {

			e.printStackTrace();
		}

		if (data == null) {
			System.out.println("valami rossz...");
		} else {
			if (data.classIndex() == -1)
				data.setClassIndex(data.numAttributes() - 1);
		}

		return data;

	}

	void calculateSMO() {

		FilteredClassifier classifier = new FilteredClassifier();
	
		SMO smoAlg = new SMO();
	
		StringToWordVector filter = new StringToWordVector();

		//Make a tokenizer
		WordTokenizer wt = new WordTokenizer();
		String delimiters = " \r\t\n.,;:\'\"()?!";
		wt.setDelimiters(delimiters);
		filter.setTokenizer(wt);
		
		
		try {
			filter.setOptions(Utils.splitOptions("-R first-last -W 1000 -prune-rate -1.0 -N 0 -stemmer weka.core.stemmers.NullStemmer -stopwords-handler weka.core.stopwords.Null -M 1"));
			filter.setInputFormat(getTrainDataSource());
			
			smoAlg.setOptions(Utils.splitOptions(
					"-C 0.18 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\""));

			classifier.setClassifier(smoAlg);
			classifier.setFilter(filter);
			classifier.buildClassifier(trainDataSource);
			File file = new File("predictions.txt");
		      // creates the file
		      file.createNewFile();
		      // creates a FileWriter Object
		      FileWriter writer = new FileWriter(file); 
			
			 for (int i = 0; i < testDataSource.numInstances(); i++) {
				   double pred = classifier.classifyInstance(testDataSource.instance(i));
				   System.out.print("ID: " + testDataSource.instance(i).value(0));
				   System.out.print(", actual: " + testDataSource.classAttribute().value((int) testDataSource.instance(i).classValue()));
				   System.out.println(", predicted: " + testDataSource.classAttribute().value((int) pred));
				   int id = (int) testDataSource.instance(i).value(0);
				   writer.write("\""+id+"\",\""+testDataSource.classAttribute().value((int) pred)+"\"\n"); 
				 }
			 
			 
		      // Writes the content to the file
		      
		      writer.flush();
		      writer.close();

			 
			 
			// Evaluation eval = new Evaluation(trainDataSource);
			// eval.crossValidateModel(classifier, trainDataSource, 10, new Random(1));
			// System.out.println(eval.toSummaryString("\nResults\n======\n", false));

			 Evaluation eval = new Evaluation(trainDataSource);
			 eval.evaluateModel(classifier, testDataSource);
			 System.out.println(eval.toSummaryString("\nResults\n======\n", false));

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public Instances getTrainDataSource() {
		return trainDataSource;
	}

	public void setTrainDataSource(Instances trainDataSource) {
		this.trainDataSource = trainDataSource;
	}

	public Instances getTestDataSource() {
		return testDataSource;
	}

	public void setTestDataSource(Instances testDataSource) {
		this.testDataSource = testDataSource;
	}

}
