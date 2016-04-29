import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Prediction {

	private DataSource trainDataSource;
	private DataSource testDataSource;

	public Prediction() {
	}

	void initDataSources() {

		trainDataSource = initDatasource("resources/train.arff");
		testDataSource = initDatasource("resources/test.arff");

	}

	DataSource initDatasource(String file) {

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

		return source;

	}

	public DataSource getTrainDataSource() {
		return trainDataSource;
	}

	public void setTrainDataSource(DataSource trainDataSource) {
		this.trainDataSource = trainDataSource;
	}

	public DataSource getTestDataSource() {
		return testDataSource;
	}

	public void setTestDataSource(DataSource testDataSource) {
		this.testDataSource = testDataSource;
	}

}
