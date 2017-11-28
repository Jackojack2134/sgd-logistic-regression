import java.lang.Math;
/**
    @author Greg Sop, Jacob Loeser
    SGD class instantiates stochastic gradient descent algorithm
*/
public class SGD {

    /**
        Instantiation of SGD where @samples is the data read in to train the
        learner randomly permuted, @epochs is the hyperparameter controlling the
        number of iterations to run SGD, @l_rate is the hyperparameter controlling
        learning rate, @dim is the dimensionality of the problem
    */
    public static double SGD(double[] samples, int epochs, double[] l_rate, int dim) {
        //Initialize constraint vector
        double[] w = new double[dim];
        for(int i = 0; i < dim; i++) {
            w[i] = 0;
        }
        //Start training in epochs
        for(int t = 0; i < epochs; t++) {
            //Have G-Oracle Produce Random Vector =>
            //Draw an example from samples and calculate logistic loss
            //Random Vector = Gradient of the Loss
            double loss = logistic_loss(w[t], samples[t]);
        }
        
    }
    //Euclidean Projection
    public static double[] project(double[] v) {
        return v;
    }
    //Logistic Loss Function
    public static double logistic_loss(int label, double[] weights, double[] data) {
        //Construct x_hat from the specifications
        double[] data_hat = data;
        //Return value of logistic loss function
        return Math.log(1+Math.exp(-label * dot(data_hat, weights)));
    }
    //Inner product
    public static double dot(double[] x, double[] w) {
        double dot = 0;
        for(int i = 0; i < x.length; i++) {
            dot += x[i] * w[i];
        }
        return dot;
    }
    public static void main(String[] args) {
        //Do file I/O to get Sample Data
    }

}
