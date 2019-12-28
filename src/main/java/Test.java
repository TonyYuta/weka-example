
import com.emaraic.ml.ModelClassifier;
import com.emaraic.ml.ModelGenerator;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Debug;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

/**
 *
 * @author Taha Emara 
 * Website: http://www.emaraic.com 
 * Email : taha@emaraic.com
 * Created on: Jul 1, 2017
 * Github link: https://github.com/emara-geek/weka-example
 */
public class Test {

   // public static final String DATASETPATH = "data/iris.2D.arff";
   public static final String DATASETPATH = "data/iris1-2-7-10_with_colors.2D.arff";
   public static final String MODElPATH = "data/model.bin";

    public static void main(String[] args) throws Exception {
        
        ModelGenerator mg = new ModelGenerator();

        Instances dataset = mg.loadDataset(DATASETPATH);

        Filter filter = new Normalize();

        // divide dataset to train dataset 80% and test dataset 20%
        int trainSize = (int) Math.round(dataset.numInstances() * 0.8);
        int testSize = dataset.numInstances() - trainSize;

        dataset.randomize(new Debug.Random(1));// if you comment this line the accuracy of the model will be droped from 96.6% to 80%
        
        //Normalize dataset
        filter.setInputFormat(dataset);
        Instances datasetnor = Filter.useFilter(dataset, filter);

        Instances traindataset = new Instances(datasetnor, 0, trainSize);
        Instances testdataset = new Instances(datasetnor, trainSize, testSize);

        // build classifier with train dataset             
        MultilayerPerceptron ann = (MultilayerPerceptron) mg.buildClassifier(traindataset);

        // Evaluate classifier with test dataset
        String evalsummary = mg.evaluateModel(ann, traindataset, testdataset);
        System.out.println("Evaluation: " + evalsummary);

        //Save model 
        mg.saveModel(ann, MODElPATH);

        //classifiy a single instance 
        ModelClassifier cls = new ModelClassifier();
        //String classname =cls.classifiy(Filter.useFilter(cls.createInstance(1.6, 0.2, 0), filter), MODElPATH);
       // String classname =cls.classifiy(Filter.useFilter(cls.createInstance(7.0, 2.1, 0), filter), MODElPATH);
        //String classname =cls.classifiy(Filter.useFilter(cls.createInstance(2.7, 2.5, 5.1, 0), filter), MODElPATH);
        //System.out.println("\n The class name for the instance with petallength = 2.7, petalwidth =2.5, and petalcolor = 5.1  is  " +classname);

        String classname =cls.classifiy(Filter.useFilter(cls.createInstance(4.7, 1.4, 1.3, 0), filter), MODElPATH);
        System.out.println("\n The class name for the instance with petallength = 4.7, petalwidth =1.4, and petalcolor = 1.3  is  " +classname);

    }

}
