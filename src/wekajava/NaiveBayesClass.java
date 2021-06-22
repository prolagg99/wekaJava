package wekajava;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

// here im using 100% for both train & test for 70% of accuracy

public class NaiveBayesClass {
    public static final String modelpath = "E:/Master/DataSet/TEST/modelNB.model";
    public static void main(String[] args) throws Exception{
        
//        DataSource source = new DataSource("E:/Master/DataSet/AccGyroMagn_Arff.arff");
        DataSource source = new DataSource("E:/Master/DataSet/TEST/datamodel3.arff");
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
        DataSource source2 = new DataSource("E:/Master/DataSet/TEST/datamodel3.arff");
        Instances testdata = source2.getDataSet();
        testdata.setClassIndex(testdata.numAttributes()-1);

        // evaluation need the testdara & traindata 
        Evaluation eval = new Evaluation(dataset);
        eval.evaluateModel(nb, testdata);
        System.out.println(eval.toSummaryString());
        
        //Save model 
        SerializationHelper.write(modelpath, nb);

        // add new instance for andorid studio prediction
//        addNewInstance(testdata, 0.358835,0.864053,-0.005573,-0.1986,-0.122358,-0.045218,-0.1986,-0.122358,-0.045218,50);
//        System.out.println("last instn testdata : \r\n " + testdata.lastInstance());
        
        // load the model 
        Classifier cls = (Classifier) weka.core.SerializationHelper.read("E:/Master/DataSet/TEST/modelNB.model");
        
        // make prediction by saved model
        for (int j = 0; j < testdata.numInstances(); j++){
            double actualClass = testdata.instance(j).classValue();
            String actual = testdata.classAttribute().value((int) actualClass);
            Instance newInst = testdata.instance(j);
//			System.out.println("actual class:"+newInst.stringValue(newInst.numAttributes()-1));
            double preNB = cls.classifyInstance(newInst);
            String predString = testdata.classAttribute().value((int) preNB);
            System.out.println("actuel : " + actual + " ,predication   " + predString);
        }
        
        // make prediction by naive bayes classifier
//        for (int j=0;j<testdata.numInstances();j++){
//                double actualClass = testdata.instance(j).classValue();
//                String actual = testdata.classAttribute().value((int) actualClass);
//                Instance newInst = testdata.instance(j);
////		System.out.println("actual class:"+newInst.stringValue(newInst.numAttributes()-1));
//                double preNB = nb.classifyInstance(newInst);
//                String predString = testdata.classAttribute().value((int) preNB);
//                System.out.println(actual+","+predString);
//        }

        // classify the new instance using the model
//        Instances labeled2 = new Instances(testdata);
//        double value = cls.classifyInstance(testdata.lastInstance());
//        labeled2.lastInstance().setClassValue(value);
//        System.out.println(value);
//        System.out.println("from the model hereeee: " + labeled2.lastInstance().stringValue(10));
//        System.out.println("last instn labeled 2 : \r\n " + labeled2.lastInstance());
                
    }
    public static void addNewInstance(Instances testdataset, double acc_x, double acc_y, double acc_z, 
            double gyro_x, double gyro_y, double gyro_z,
            double magn_x, double magn_y, double magn_z, int steps ){
        
        Instance inst = new Instance(11);
        inst.setValue(testdataset.attribute(0), acc_x);
        inst.setValue(testdataset.attribute(1), acc_y);
        inst.setValue(testdataset.attribute(2), acc_z);
        inst.setValue(testdataset.attribute(3), gyro_x);
        inst.setValue(testdataset.attribute(4), gyro_y);
        inst.setValue(testdataset.attribute(5), gyro_z);
        inst.setValue(testdataset.attribute(6), magn_x);
        inst.setValue(testdataset.attribute(7), magn_y);
        inst.setValue(testdataset.attribute(8), magn_z);
        inst.setValue(testdataset.attribute(9), steps);

        // add
        testdataset.add(inst);
    }
}