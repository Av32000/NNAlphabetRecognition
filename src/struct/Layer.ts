import { ActivationFunction } from './NeuralNetwork';

export class Layer {
	weight: number[][] = [];
	biases: number[] = [];
	activationFunction: ActivationFunction;
	constructor(
		inputCount: number,
		outputCount: number,
		activationFunction: ActivationFunction,
	) {
		// Setup weight
		this.weight = new Array<number[]>(inputCount);
		for (let i = 0; i < inputCount; i++) {
			this.weight[i] = new Array<number>(outputCount).fill(0);
		}

		// Setup Biases
		this.biases = new Array<number>(outputCount).fill(0);
		this.activationFunction = activationFunction;
	}

	CalculateOutput(inputValues: number[]) {
		if (inputValues.length != this.weight.length) {
			throw new Error(
				`Invalid Inputs Size :\nExpexted : ${this.weight.length}\nReceived : ${inputValues.length}`,
			);
		}

		let result = new Array<number>(this.biases.length);

		for (let outputIndex = 0; outputIndex < this.biases.length; outputIndex++) {
			let nodeValue = this.biases[outputIndex];

			for (let inputIndex = 0; inputIndex < inputValues.length; inputIndex++) {
				const inputValue = inputValues[inputIndex];
				nodeValue += inputValue * this.weight[inputIndex][outputIndex];
			}

			result[outputIndex] = this.ActivateOutput(nodeValue);
		}

		return result;
	}

	ActivateOutput(output: number) {
		switch (this.activationFunction) {
			case 'BinaryStep':
				return output < 0 ? 0 : 1;
			case 'Sigmoid':
				return 1 / (1 + Math.exp(-output));
			case 'ReLU':
				return (
					(Math.exp(output) - Math.exp(-output)) /
					(Math.exp(output) + Math.exp(-output))
				);
		}
	}

	NodeCost(output: number, expectedOutput: number) {
		const error = output - expectedOutput;
		return error * error;
	}
}
