package wekajava;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Debug;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

// here im using 80% train 
// & 20% for test for 76% of accuracy

public class NNsClass {
    public static final String modelpath = "E:/Master/DataSet/OurDataSet/OurDataSetModelNNS.model";
    public static void main(String[] args) throws Exception {
//        DataSource source = new DataSource("E:/Master/DataSet/AccGyroMagn_Arff.arff");
        DataSource source = new DataSource("E:/Master/DataSet/OurDataSet/OurDataSetArff.arff");
        Instances dataset = source.getDataSet();
        dataset.setClassIndex(dataset.numAttributes()-1);
        int numClasses = dataset.numClasses();
        System.out.println(numClasses);
        for (int i=0;i<numClasses;i++){
                String classValue = dataset.classAttribute().value(i);
                System.out.println("the " + i + "th class value : " + classValue);
        }

        dataset.randomize(new Debug.Random(1));// if you comment this line the accuracy of the model will be droped from 96.6% to 80%

        //Normalize dataset
        Normalize normalize = new Normalize();
        normalize.setInputFormat(dataset);
        Instances newdata = Filter.useFilter(dataset, normalize);
//        
        // divide dataset to train dataset 80% and test dataset 20%
//        int trainSize = (int) Math.round(dataset.numInstances() * 0.8);
//        int testSize = dataset.numInstances() - trainSize;
//
//        Instances traindataset = new Instances(newdata, 0, trainSize);
//        Instances testdataset = new Instances(newdata, trainSize, testSize);
////        

//        load test data
        DataSource source2 = new DataSource("E:/Master/DataSet/OurDataSet/OurDataSetArff.arff");
        Instances testdata = source2.getDataSet();
        testdata.setClassIndex(testdata.numAttributes()-1);


        // build classifier with train dataset   
        // neural ntworks classifier
        MultilayerPerceptron ann = new MultilayerPerceptron();
        ann.buildClassifier(dataset);
        
        // Evaluate classifier with test dataset
        // evaluation need the testdara & traindata 
        Evaluation eval = new Evaluation(dataset);
        eval.evaluateModel(ann, testdata);
        System.out.println(eval.toSummaryString());
        
        //Save model 
        SerializationHelper.write(modelpath, ann);

        // make prediction by neural networks classifier
//        for (int j = 0; j < testdataset.numInstances(); j++){
//                double actualClass = testdataset.instance(j).classValue();
//                String actual = testdataset.classAttribute().value((int) actualClass);
//                Instance newInst = testdataset.instance(j);
////			System.out.println("actual class:"+newInst.stringValue(newInst.numAttributes()-1));
//                double preNN = ann.classifyInstance(newInst);
//                String predString = testdataset.classAttribute().value((int) preNN);
//                System.out.println("actuel : " + actual + " ,predication   " + predString);
//        }


//        addNewInstance(testdata, 0.358835,0.864053,-0.005573,-0.1986,-0.122358,-0.045218,-0.1986,-0.122358,-0.045218,50);
//        System.out.println("last instn testdata : \r\n " + testdata.lastInstance());

        // make prediction by model of NNS
        Classifier cls = (Classifier) weka.core.SerializationHelper.read("E:/Master/DataSet/OurDataSet/modelNNS.model");
        
        // classify the new instance using the model
//        Instances labeled2 = new Instances(testdata);
//        double value = cls.classifyInstance(testdata.lastInstance());
//        labeled2.lastInstance().setClassValue(value);
//        System.out.println(value);
//        System.out.println("from the model hereeee: " + labeled2.lastInstance().stringValue(10));
//        System.out.println("last instn labeled 2 : \r\n " + labeled2.lastInstance());
//        

        
        // classify the testdata using the model
          for (int j = 0; j < testdata.numInstances(); j++){
                double actualClass = testdata.instance(j).classValue();
                String actual = testdata.classAttribute().value((int) actualClass);
                Instance newInst = testdata.instance(j);
//			System.out.println("actual class:"+newInst.stringValue(newInst.numAttributes()-1));
                double preNN = cls.classifyInstance(newInst);
                String predString = testdata.classAttribute().value((int) preNN);
                System.out.println("actuel : " + actual + " ,predication   " + predString);
        }

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