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
        
        TODO: Data types may not be correct. For example return type should probably
        be double array since we're working with parameter vectors.
        TODO: Check to see if this gradient is correct analytically.
        Source: https://stats.stackexchange.com/questions/219241/gradient-for-logistic-loss-function
        
    */
    public static double SGD(double[][] samples, int epochs, double l_rate, int dim) {
        //Initialize constraint vector
        double[] w = new double[dim];
        for(int i = 0; i < dim; i++) {
            w[i] = 0;
        }
        //Start training in epochs
        for(int t = 0; t < epochs; t++) {
            //Have G-Oracle Produce Random Vector =>
            //Draw an example from samples and calculate gradient of logistic loss
            
            //Evaluate loss function at t(th) weight and fresh example
            double loss = logistic_loss(w[t], samples[t]);
            //Gradient of the loss is difference between true label and loss
            //This assumes that true label lies at 1st entry in each row of sample
            double grad_loss = loss - samples[t][0];
            
            //Update Step
            double[] value = update(w[t], loss, l_rate);
            //Projection Step
            w[t+1] = project(value, w);
        }
        //Return Empirical Loss
        double w_hat = 0;
        for(int t = 0; t < epochs; t++) {
            w_hat += w[t];
        }
        return (w_hat/epochs);
        
    }
    //Loss Update
    public static double update(double[] w_t, double err, double l_rate) {
        return w_t-(l_rate*err);
    }
    //TODO: Finish Projection Function
    //Euclidean Projection
    public static double project(double v, double[] w) {
        double min = v - w[0];
        for (int t = 1; t < samples.length; t++) {
            if (v - w[t] < min) {
                min = v - w[t];
            }
        }
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
    //TODO: Test with actual data
    public static void main(String[] args) {
        //Do file I/O to get Sample Data
    }

}
