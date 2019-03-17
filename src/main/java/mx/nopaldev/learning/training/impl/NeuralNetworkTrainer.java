package mx.nopaldev.learning.training.impl;

import static java.lang.System.out;
import static java.lang.Math.random;


public class NeuralNetworkTrainer
{
	private double[] weights;
	private double bias;
	private double learningRate = 0.1;
	private int iterations = 100000;

	public NeuralNetworkTrainer()
	{
		this.weights = new double[] { random(), random() };
		this.bias = random();
	}

	public NeuralNetworkTrainer train(double[][] trainingData)
	{
		for (int iteration = 0; iteration < iterations; iteration++)
		{
			for (int pIndex = 0; pIndex < trainingData.length; pIndex++)
			{
				double[] point = trainingData[pIndex];
				double z = sumWeights(point);

				double prediction = sigmoid(z);
				double yValue = point[point.length - 1];

				// derivatives
				double derivativeCostPred = (prediction - yValue) * 2;
				double derivativeForSigmoid = sigmoid(z) * (1 - sigmoid(z));

				double derivativeFactor = derivativeCostPred * derivativeForSigmoid;

				int dependantSize = point.length - 1;

				// update the weights
				for (int vIndex = 0; vIndex < dependantSize; vIndex++)
				{
					weights[vIndex] -= learningRate * derivativeFactor * point[vIndex];
				}

				// update the bias as well
				bias = learningRate * derivativeFactor * 1;
			}
		}

		return this;
	}

	protected double sumWeights(double[] values)
	{
		double prediction = 0;
		int dependantVariablesLength = values.length - 1;
		for (int index = 0; index < dependantVariablesLength; index++)
		{
			prediction += values[index] * weights[index];
		}

		prediction += bias;

		return prediction;
	}

	public double predict(double[] values)
	{
		return sigmoid(sumWeights(values));
	}

	protected double sigmoid(double x)
	{
		return 1 / (1 + Math.exp(-x));
	}

	public static void main(String[] args)
	{
		double[][] training = {
				// blue ones
				{ 2, 1, 0 },
				{ 3, 1, 0 },
				{ 2, .5, 0 },
				{ 1, 1, 0 },

				// red ones
				{ 3, 1.5, 1 },
				{ 3.5, .5, 1 },
				{ 4, 1.5, 1 },
				{ 5.5, 1, 1 },
		};

		// create and train the model directly
		final NeuralNetworkTrainer model =
				new NeuralNetworkTrainer()
						.train(training);

		// predict an observation
		double classPrediction =
				model.predict(new double[] { 1, 1 });
		out.println("Model predicted " + classPrediction);
	}
}
