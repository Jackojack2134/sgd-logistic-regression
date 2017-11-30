import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.lang.Math;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
/**
    @author Greg Sop, Jacob Loeser
    SGD class instantiates stochastic gradient descent algorithm
*/
public class SGD {
    /**
	 * Determines if the vector of coordinates, @code(u), is within the hypersphere of dimension
	 * @code(u.length) and radius of @code(radius); i.e. determines if a point is within
	 * the hypersphere.
	 * 
	 * @param u
	 * 		the vector of coordinates for a single point
	 * @return
	 * 		true if the point lies within the hypersphere
	 */
	private static double hyperSphereMagnitude(double[] u) {
		double magnitude;
		
		double squaredRadius = 0;
		
		// Get the sum of the square of every coordinate value.
		// i.e. a^2 + b^2 + ...
		for (int i = 0; i < u.length; i++) {
			squaredRadius += u[i] * u[i];
		}
		
		// Get the "radius" to the point and compare it to the actual radius.
		// i.e. r = sqrt(a^2 + b^2 + ...)
		magnitude = Math.sqrt(squaredRadius);
		
		return magnitude;
	}
    private static double[] testing(double[] w, double[][] sample) {
        //Run weights on each example in sample
        double[] results = new double[2];
        double[] losses = new double[sample.length];
        double[] data = new double[w.length];
        int correct = 0;
        int label = (int) sample[0][0];
        losses[0] = logistic_loss(label, w, sample[0]);
        double min = losses[0];
        for(int i = 1; i < sample.length; i++) {
            label = (int) sample[i][0];
            losses[i] = logistic_loss(label, w, sample[i]);
            if(losses[i] < min) {
                min = losses[i];
            }
            for(int j = 1; j < data.length; j++) {
                data[j-1] = sample[i][j];
            }
            data[w.length - 1] = 1;
            if(sample[i][0]*dot(w,data) > 0) {
                correct++;
            }
        }
        double mean = 0;
        for (int i = 0; i < losses.length; i++) {
            mean += losses[i];
        }
        mean /= losses.length;
        results[0] = mean;
        
        double c_error = 1.0 - (1.0 * correct / sample.length);
        results[1] = c_error;
        return results;
    }
    
    /**
	 * Determines if the vector of coordinates, @code(u), is within the hypercube of
	 * dimension @code(u.length) and side length of 2 * @code(radius); i.e. determines
	 * if a point is within the hypercube.
	 * 
	 * @param u
	 * 		the vector of coordinates for a single point
	 * @return
	 * 		true if the point lies within the hypercube
	 */
	private static boolean isHyperCube(double[] u) {
		boolean isCube = true;
		
		/*
		 * Compares each coordinate to the range of restriction in each dimension.
		 * Short-circuits if any coordinate lies outside the range, otherwise iterates
		 * through every point.
		 */
		int i = 0;
		while (isCube && i < u.length ) {
			isCube = u[i] >= -1 && u[i] <= 1;
			i++;
		}
		
		return isCube;
	}

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
    public static double[] SGD(double[][] samples, int epochs, double l_rate, int dim, int scenario) {
        //Initialize weight vector
        double[][] w = new double[epochs][dim];
        double[] w_hat = new double[dim];
        for(int i = 0; i < epochs; i++) {
            for(int j = 0; j < dim; j++) {
                w[i][j] = 0;
            }
        }
        for(int i = 0; i < dim; i++) {
            w_hat[i] = 0;
        }
        //Start training in epochs
        for(int t = 1; t < epochs; t++) {
            //Have G-Oracle Produce Random Vector =>
            //Draw an example from samples and calculate gradient of logistic loss
            
            //Evaluate loss function at t(th) weight and fresh example
            int label = (int) samples[t][0];
	        double loss = logistic_loss(label, w[t], samples[t]);
            //Gradient of the loss is difference between true label and loss
            //This assumes that true label lies at 1st entry in each row of sample
            double grad_loss = loss - label;
            
            //Update Step
            double value = update(grad_loss, l_rate);
            //Projection Step
            w[t] = project(w[t-1], value, scenario);
        }
        //Return Empirical Loss
        for (int i = 0; i < dim; i++) {
            for(int t = 0; t < epochs; t++) {
                w_hat[i] += w[t][i];
            }
            w_hat[i] /= epochs;
        }
        return w_hat;
    }
    
    //Loss Update
    public static double update(double err, double l_rate) {
        return err * l_rate;
    }
    
    //TODO: Finish Projection Function
    //Euclidean Projection
    public static double[] project(double[] v, double value, int scenario) {
        // Updating with the learning rate * gradient of the loss.
        for (int i = 0; i < v.length; i++) {
            v[i] -= value;
        }
        
        double[] w = new double[v.length];
        // Scenario for the hypercube.
        if (scenario == 0) {
            // Check if within the hypercube.
            if (!isHyperCube(v)) {
                for(int i = 0; i < v.length; i++) {
                    if(v[i] > 1) {
                        double step = v[i] - 1;
                        w[i] = v[i] - step;
                    } else if(v[i] < -1) {
                        double step = v[i] + 1;
                        w[i] = v[i] - step;
                    }
                }
            } else {
                w = v;
            }
        } else if (scenario == 1) {     // Otherwise, scenario for the hypersphere.
            double magnitude = hyperSphereMagnitude(v);
            // Check if within the hypersphere.
            if (magnitude > 1) {
                for (int i = 0; i < v.length; i++) {
                    w[i] = v[i] / magnitude;
                }
            } else {
                w = v;
            }
        }
        return w;
    }
    
    //Logistic Loss Function
    public static double logistic_loss(int label, double[] weights, double[] data) {
        //Construct x_hat from the specifications
        double[] data_hat = new double[weights.length];
        for (int i = 1; i < data.length; i++) {
            data_hat[i - 1] = data[i];
        }
        data_hat[weights.length - 1] = 1;
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
        // The output file. Try to create/open it and exit on failure.
	PrintWriter outFile = null;
	try {
		outFile = new PrintWriter("stats.txt", "UTF-8");
	} catch (FileNotFoundException e) {
		System.err.println("Output file does not exist and could not be created.");
		System.exit(0);
	} catch (UnsupportedEncodingException e) {
		System.err.println("Encoding type is unsupported.");
		System.exit(0);
        }
        
        // Dim-size is 6.
        int dim = 6;
        int numSamples;
        int scenario;
        double l_rate = 0.02;
        
        if (args.length != 4) {
            System.err.println("Incorrect number of arguments. Enter the input file, the output file, number of samples, and the scenario number.");
            System.exit(0);
        }
        
        String fileName = args[0];
        String testFileName = args[1];
	    numSamples = Integer.parseInt(args[2]);
        scenario = Integer.parseInt(args[3]);
        BufferedReader br = null;
        FileReader fr = null;
        double[][] results = new double[30][2];
        int numTestExamples = 400;
        double[][] testData = new double[numTestExamples][dim];
        
        try {
            br = new BufferedReader(new FileReader(testFileName));
            String currentLine = br.readLine();
            int count = 0;
            while (currentLine != null) {
                String[] strArr = currentLine.split("\\s+");
                for (int i = 0; i < strArr.length; i++) {
                    testData[count][i] = Double.parseDouble(strArr[i]);
                }
                count++;
            }
	} catch (IOException e) {
            System.err.println("Error reading from " + testFileName);
            System.exit(0);
	} finally {
            try {
                    if (br != null) {
                        br.close();
                    }

                    if (fr != null) {
                        fr.close();
                    }
                } catch (IOException ex) {
                    System.err.println("Error while closing file.");
                }
            
            try {
                br = new BufferedReader(new FileReader(fileName));

                String currentLine = br.readLine();
                int i, testNum;
                testNum = 0;
                double[][] samples = new double[numSamples][dim];
                while (currentLine != null) {
                    i = 0;
                    while (i < numSamples) {
                        String[] strArr = currentLine.split("\\s+");
                        for (int j = 0; j < strArr.length; j++) {
                            samples[i][j] = Double.parseDouble(strArr[j]);
                        }
                        i++;
                        currentLine = br.readLine();
                    }
                    double[] sgdResult = SGD(samples, numSamples, l_rate, dim, scenario);
                    double[] testResult = testing(sgdResult, testData); 
                    for (int j = 0; j < testResult.length; j++) {
                        results[testNum] = testResult;
                    }
                }
                double lossMean = results[0][0];
                double lossMin = results[0][0];
                double classMean = results[0][1];
                
                for (int j = 1; j < results.length; j++) {
                    lossMean += results[j][0];
                    classMean += results[j][1];
                    if (results[j][0] < lossMin) {
                        lossMin = results[j][0];
                    }
                }
                
                lossMean /= results.length;
                classMean /= results.length;
                
                double lossStdDev = 0;
                double classStdDev = 0;
                for (int j = 0; j < results.length; j++) {
                    lossStdDev += Math.pow(results[j][0] - lossMean, 2);
                    classStdDev += Math.pow(results[j][1] - classMean, 2);
                }
                lossStdDev /= results.length - 1;
                lossStdDev = Math.sqrt(lossStdDev);
                classStdDev /= results.length - 1;
                classStdDev = Math.sqrt(classStdDev);
                
                double excessRisk = lossMean - lossMin;
		    
                outFile.println(lossMean);
                outFile.println(lossStdDev);
                outFile.println(lossMin);
                outFile.println(excessRisk);
                outFile.println(classMean);
                outFile.println(classStdDev);
            } catch (IOException e) {
                System.err.println("Error reading from: " + fileName);
            } finally {
                try {
                    if (br != null) {
                        br.close();
                    }

                    if (fr != null) {
                        fr.close();
                    }
                } catch (IOException ex) {
                    System.err.println("Error while closing file.");
                }
            }
        outFile.close();
    }
}
}
