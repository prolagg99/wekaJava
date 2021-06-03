package wekajava;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Debug;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

// here im using 100% train 
// & 100% for test for 78% of accuracy

public class NNsClass {
    public static final String modelpath = "E:/Master/DataSet/NNSmodel.model";
    public static void main(String[] args) throws Exception {
//        DataSource source = new DataSource("E:/Master/DataSet/AccGyroMagn_Arff.arff");
        DataSource source = new DataSource("E:/Master/DataSet/TEST/test.arff");
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
//        

//        load test data
        DataSource source2 = new DataSource("E:/Master/DataSet/TEST/test.arff");
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
        for (int j = 0; j < testdata.numInstances(); j++){
                double actualClass = testdata.instance(j).classValue();
                String actual = testdata.classAttribute().value((int) actualClass);
                Instance newInst = testdata.instance(j);
//			System.out.println("actual class:"+newInst.stringValue(newInst.numAttributes()-1));
                double preNN = ann.classifyInstance(newInst);
                String predString = testdata.classAttribute().value((int) preNN);
                System.out.println("actuel : " + actual + " ,predication   " + predString);
        }
    }
}