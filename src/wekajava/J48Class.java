package wekajava;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import static wekajava.WekaJava.modelpath;

// here im using 100% train 
// & 20% for test for 89% of accuracy

public class J48Class {
    public static final String modelpath = "E:/Master/DataSet/J48model.model";
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
        
//        //Normalize dataset
//        Normalize normalize = new Normalize();
//        normalize.setInputFormat(dataset);
//        Instances newdata = Filter.useFilter(dataset, normalize);
        
        // divide dataset to train dataset 80% and test dataset 20%
        int trainSize = (int) Math.round(dataset.numInstances() * 0.8);
        int testSize = dataset.numInstances() - trainSize;
        
//        Instances traindataset = new Instances(dataset, 0, trainSize);
        Instances testdataset = new Instances(dataset, trainSize, testSize);
        
        // decission tree classifier	
        J48 tree = new J48();
        tree.buildClassifier(dataset);
        
//        // load test data
//        DataSource source2 = new DataSource("E:/Master/DataSet/TEST/test.arff");
//        Instances testdata = source2.getDataSet();
//        testdata.setClassIndex(testdata.numAttributes()-1);

        // evaluation need the testdara & traindata 
        Evaluation eval = new Evaluation(dataset);
        eval.evaluateModel(tree, testdataset);
        System.out.println(eval.toSummaryString());
        
        //Save model 
        SerializationHelper.write(modelpath, tree);

        // make prediction by naive bayes classifier
//        for (int j=0;j<testdataset.numInstances();j++){
//            double actualClass = testdataset.instance(j).classValue();
//            String actual = testdataset.classAttribute().value((int) actualClass);
//            Instance newInst = testdataset.instance(j);
////          System.out.println("actual class:"+newInst.stringValue(newInst.numAttributes()-1));
//            double preNB = tree.classifyInstance(newInst);
//            String predString = testdataset.classAttribute().value((int) preNB);
//            System.out.println(actual+","+predString);
//        }


        // -0.086055,-0.361837,-0.64796,0.057196,0.075941,0.129704,-282.665838,264.22574,147.078061,walking --> walking
        // 0.004441,-0.006493,0.998635,-0.00011,0.000041,0.000714,-241.053011,195.015964,26.104802,sitting --> sitting
        //0.358835,0.864053,-0.005573,-0.1986,-0.122358,-0.045218,-0.1986,-0.122358,-0.045218,running --> standding
        
        // add a new instance to which we apply a prediction 
        addNewInstance(testdataset, -0.086055,-0.361837,-0.64796,0.057196,0.075941,0.129704,-282.665838,264.22574,147.078061);
        System.out.println("last instn testdataset : \r\n " + testdataset.lastInstance());  
        
        // make prediction of a new instance using saved model 
        // load the model
        Classifier cls = (Classifier) weka.core.SerializationHelper.read("E:/Master/DataSet/J48model.model");
        
        Instances labeled2 = new Instances(testdataset);
        double value = cls.classifyInstance(testdataset.lastInstance());
        labeled2.lastInstance().setClassValue(value);
        System.out.println(value);
        System.out.println("from the model hereeee: " + labeled2.lastInstance().stringValue(9));
        System.out.println("last instn labeled 2 : \r\n " + labeled2.lastInstance());
    }
    
     public static void addNewInstance(Instances testdataset, double acc_x, double acc_y, double acc_z, 
            double gyro_x, double gyro_y, double gyro_z,
            double magn_x, double magn_y, double magn_z ){
        
        Instance inst = new Instance(10);
        inst.setValue(testdataset.attribute(0), acc_x);
        inst.setValue(testdataset.attribute(1), acc_y);
        inst.setValue(testdataset.attribute(2), acc_z);
        inst.setValue(testdataset.attribute(3), gyro_x);
        inst.setValue(testdataset.attribute(4), gyro_y);
        inst.setValue(testdataset.attribute(5), gyro_z);
        inst.setValue(testdataset.attribute(6), magn_x);
        inst.setValue(testdataset.attribute(7), magn_y);
        inst.setValue(testdataset.attribute(8), magn_z);

        // add
        testdataset.add(inst);
    }
}