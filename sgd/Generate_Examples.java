package sgd;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.InputMismatchException;
import java.util.Random;
import java.util.Scanner;

public class Generate_Examples {
	/**
	 * The pseudo-random number generator.
	 */
	static Random rand = new Random();
	
	/**
	 * Determines if the vector of coordinates, @code(u), is within the hypersphere of dimension
	 * @code(u.length) and radius of @code(radius); i.e. determines if a point is within
	 * the hypersphere.
	 * 
	 * @param u
	 * 		the vector of coordinates for a single point
	 * @param radius
	 * 		the radius of the hypersphere
	 * @return
	 * 		true if the point lies within the hypersphere
	 */
	private static boolean isHyperSphere(double[] u, double radius) {
		boolean isSphere;
		
		double squaredRadius = 0;
		
		// Get the sum of the square of every coordinate value.
		// i.e. a^2 + b^2 + ...
		for (int i = 0; i < u.length; i++) {
			squaredRadius += u[i] * u[i];
		}
		
		// Get the "radius" to the point and compare it to the actual radius.
		// i.e. r = sqrt(a^2 + b^2 + ...)
		isSphere = Math.sqrt(squaredRadius) <= radius;
		
		return isSphere;
	}
	
	/**
	 * Determines if the vector of coordinates, @code(u), is within the hypercube of
	 * dimension @code(u.length) and side length of 2 * @code(radius); i.e. determines
	 * if a point is within the hypercube.
	 * 
	 * @param u
	 * 		the vector of coordinates for a single point
	 * @param radius
	 * 		half of the side length
	 * @return
	 * 		true if the point lies within the hypercube
	 */
	private static boolean isHyperCube(double[] u, double radius) {
		boolean isCube = true;
		
		/*
		 * Compares each coordinate to the range of restriction in each dimension.
		 * Short-circuits if any coordinate lies outside the range, otherwise iterates
		 * through every point.
		 */
		int i = 0;
		while (isCube && i < u.length ) {
			isCube = u[i] >= -radius && u[i] <= radius;
			i++;
		}
		
		return isCube;
	}
	
	/**
	 * Returns a Gaussian vector of size @code(dimensions) with mean @code(mean) and variance
	 * @code(variance).
	 * 
	 * @param dimensions
	 * @param mean
	 * @param variance
	 * @return
	 */
	private static double[] getNormalDistribution(int dimensions, double mean, double variance) {
		// Fill the vector with a value pulled from the desired Gaussian distribution.
		double[] u = new double[dimensions];
		for (int i = 0; i < dimensions; i++) {
			u[i] = rand.nextGaussian() * variance + mean;
		}
		return u;
	}

	/**
	 *  Creates a file filled with examples to be used for Stochastic Gradient Descent.
	 * @param args
	 * 		the input arguments
	 */
	public static void main(String[] args) {
		// The input stream.
		Scanner in = new Scanner(System.in);
		
		// The output file. Try to create/open it and exit on failure.
		PrintWriter outFile = null;
		try {
			outFile = new PrintWriter("sdg_examples.txt", "UTF-8");
		} catch (FileNotFoundException e) {
			System.err.println("Output file does not exist and could not be created.");
			System.exit(0);
		} catch (UnsupportedEncodingException e) {
			System.err.println("Encoding type is unsupported.");
			System.exit(0);
		}
		
		// Ask for the dimensionality. If an incorrect input, then exit.
		System.out.print("Please enter the number of dimensions: ");
		int dimensions = 1;
		try {
			dimensions = in.nextInt();
			while (dimensions <= 0) {
				System.out.println("Value must be > 0.");
				dimensions = in.nextInt();
			}
		} catch (InputMismatchException e) {
			System.err.println("Incorrect input. Must be a positive integer.");
			System.exit(0);
		}
		
		// Ask for the sigma (variance = sigma^2). If an incorrect input, then exit.
		System.out.print("Please enter a sigma value: ");
		double sigma = 0.05;
		try {
			sigma = in.nextDouble();
		} catch (InputMismatchException e) {
			System.err.println("Incorrect input. Input must be a number.");
			System.exit(0);
		}
		
		// Variance = sigma^2
		double variance = sigma * sigma;
		double mean;
		// Radius of the hypersphere and hypercube. (1.0 is the unit value)
		double radius = 1.0;
		int y;
		
		// Determines whether the domain is a hypercube (1) or hypersphere (2).
		System.out.print("Please input a scenario (1 or 2): ");
		int scenario = 1;
		try {
			scenario = in.nextInt();
			while (!(scenario == 1 || scenario == 2)) {
				System.out.println("The scenario can only be 1 or 2.");
				scenario = in.nextInt();
			}
		} catch (InputMismatchException e) {
			System.err.println("Incorrect input. Input must be an integer.");
			System.exit(0);
		}
		
		// Asks for the number of examples to create and place in the file.
		System.out.print("Enter the number of examples to create: ");
		int n = 0;
		try {
			n = in.nextInt();
			while (n <= 0) {
				System.out.println("There must be at least 1 example.");
				n = in.nextInt();
			}
		} catch (InputMismatchException e) {
			System.err.println("Incorrect input. Input must be an integer.");
			System.exit(0);
		}

		double[] u;
		// How many times to create the n examples. (30 by default)
		// If > 1, then there will be repeatNum * n + (repeatNum - 1) filled lines.
		// This is due to the example set division lines.
		int repeatNum = 30;
		for (int i = 0; i < repeatNum; i++) {
			for (int j = 0; j < n; j++) {
				
				// Determine the two sets of 1/2 probabilities.
				if (rand.nextFloat() < 0.5) {
					y = -1;
				} else {
					y = 1;
				}
				// The mean is 1 / # of elements and whose positivity is decided by y.
				mean = y * (1.0 / dimensions);
				
				// Get the distribution.
				u = getNormalDistribution(dimensions, mean, variance);
				
				// Ensure that the vector's values lie within the specified domain.
				switch (scenario) {
				case 1:
					while (!isHyperCube(u, radius)) {
						u = getNormalDistribution(dimensions, mean, variance);
					}
					break;
				case 2:
					while (!isHyperSphere(u, radius)) {
						u = getNormalDistribution(dimensions, mean, variance);
					}
					break;
				default:
					System.err.println("There are only two scenarios: 1 and 2");
					System.exit(0);
					break;
				}
				
				// Print the example (y-value and vector values) to the output file.
				outFile.print(y + " ");
				for (int k = 0; k < u.length - 1; k++) {
					outFile.print(u[k] + " ");
				}
				outFile.println(u[u.length - 1]);
			}
			// If not the last example set, divide each set with a unique phrase.
			if (i != repeatNum - 1) {
				outFile.println("end_example");
			}
		}
		
		// Close output and input streams.
		outFile.close();
		in.close();
	}
}
