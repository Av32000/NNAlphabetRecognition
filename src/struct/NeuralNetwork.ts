import { Dataset } from './Dataset';
import { Layer } from './Layer';

export type ActivationFunction = 'BinaryStep' | 'Sigmoid' | 'ReLU';

export class NeuralNetwork {
	layers: Layer[] = [];
	constructor(layers: number[], activationFunction: ActivationFunction) {
		// Setup Layers
		for (let i = 0; i < layers.length; i++) {
			const element = layers[i];
			if (i > 0) {
				this.layers.push(
					new Layer(layers[i - 1], layers[i], activationFunction),
				);
			}
		}
	}

	Learn(dataset: Dataset, learnRate: number) {
		const h = 0.0001;
		const originalCost = this.DatasetCost(dataset);

		this.layers.forEach(layer => {
			for (let nodeInput = 0; nodeInput < layer.inputCount; nodeInput++) {
				for (let nodeOutput = 0; nodeOutput < layer.outputCount; nodeOutput++) {
					layer.weight[nodeInput][nodeOutput] += h;
					const deltaCost = this.DatasetCost(dataset) - originalCost;
					layer.weight[nodeInput][nodeOutput] -= h;
					layer.costGradientWeight[nodeInput][nodeOutput] = deltaCost / h;
				}
			}

			for (let biasIndex = 0; biasIndex < layer.biases.length; biasIndex++) {
				layer.biases[biasIndex] += h;
				const deltaCost = this.DatasetCost(dataset) - originalCost;
				layer.biases[biasIndex] -= h;
				layer.costGradientBiases[biasIndex] = deltaCost / h;
			}
		});

		this.layers.forEach(layer => {
			layer.ApplyGradients(learnRate);
		});
	}

	CalculateOutputs(inputValues: number[]) {
		let currentResult = inputValues;

		this.layers.forEach(layer => {
			currentResult = layer.CalculateOutput(currentResult);
		});

		return currentResult;
	}

	Cost(inputValues: number[], expectedOutputs: number[]) {
		const outputs = this.CalculateOutputs(inputValues);
		const outputLayer = this.layers[this.layers.length - 1];
		let result = 0;

		for (let i = 0; i < outputs.length; i++) {
			result += outputLayer.NodeCost(outputs[i], expectedOutputs[i]);
		}

		return result;
	}

	DatasetCost(dataset: Dataset) {
		let result = 0;
		dataset.elements.forEach(e => {
			result += this.Cost(e.inputs, e.expectedOutputs);
		});

		return result / dataset.elements.length;
	}
}
