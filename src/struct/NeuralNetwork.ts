import { Layer } from './Layer';

export type ActivationFunction = 'BinaryStep' | 'Sigmoid' | 'ReLU'

export class NeuralNetwork {
	layers: Layer[] = [];
	constructor(layers: number[], activationFunction: ActivationFunction) {
		// Setup Layers
		for (let i = 0; i < layers.length; i++) {
			const element = layers[i];
			if (i > 0) {
				this.layers.push(new Layer(layers[i - 1], layers[i], activationFunction));
			}
		}
	}

	CalculateOutputs(inputValues: number[]) {
		let currentResult = inputValues;

		this.layers.forEach(layer => {
			currentResult = layer.CalculateOutput(currentResult);
		});

		return currentResult;
	}
}
