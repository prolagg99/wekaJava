package wekajava;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import static wekajava.J48Class.modelpath;

// here im using 100% for both train & test for 70% of accuracy

public class NaiveBayesClass {
    public static final String modelpath = "E:/Master/DataSet/NBmodel.model";
    public static void main(String[] args) throws Exception{
        
//        DataSource source = new DataSource("E:/Master/DataSet/AccGyroMagn_Arff.arff");
        DataSource source = new DataSource("E:/Master/DataSet/TEST/test.arff");
        Instances dataset = source.getDataSet();
        dataset.setClassIndex(dataset.numAttributes()-1);
        int numClasses = dataset.numClasses();
        for (int i=0;i<numClasses;i++){
                String classValue = dataset.classAttribute().value(i);
                System.out.println("the "+i+"th class value:"+classValue);
        }
        
//          // divide dataset to train dataset 80% and test dataset 20%
//        int trainSize = (int) Math.round(dataset.numInstances() * 0.8);
//        int testSize = dataset.numInstances() - trainSize;
//        
////        Instances traindataset = new Instances(dataset, 0, trainSize);
//        Instances testdataset = new Instances(dataset, trainSize, testSize);
        
        // naive bayes classifier	
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(dataset);
        
        // load test data
        DataSource source2 = new DataSource("E:/Master/DataSet/TEST/test.arff");
        Instances testdata = source2.getDataSet();
        testdata.setClassIndex(testdata.numAttributes()-1);

        // evaluation need the testdara & traindata 
        Evaluation eval = new Evaluation(dataset);
        eval.evaluateModel(nb, testdata);
        System.out.println(eval.toSummaryString());
        
        //Save model 
        SerializationHelper.write(modelpath, nb);

        // make prediction by naive bayes classifier
        for (int j=0;j<testdata.numInstances();j++){
                double actualClass = testdata.instance(j).classValue();
                String actual = testdata.classAttribute().value((int) actualClass);
                Instance newInst = testdata.instance(j);
//		System.out.println("actual class:"+newInst.stringValue(newInst.numAttributes()-1));
                double preNB = nb.classifyInstance(newInst);
                String predString = testdata.classAttribute().value((int) preNB);
                System.out.println(actual+","+predString);
        }
    }
}