import { ActivationFunction } from './NeuralNetwork';

export class Layer {
	weight: number[][] = [];
	biases: number[] = [];
	activationFunction: ActivationFunction;

	inputCount: number;
	outputCount: number;

	costGradientWeight: number[][] = [];
	costGradientBiases: number[] = [];
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

		this.costGradientWeight = new Array<number[]>(inputCount);
		for (let i = 0; i < inputCount; i++) {
			this.costGradientWeight[i] = new Array<number>(outputCount).fill(0);
		}

		// Setup Biases
		this.biases = new Array<number>(outputCount).fill(0);
		this.costGradientBiases = new Array<number>(outputCount).fill(0);

		// Save Parameters
		this.inputCount = inputCount;
		this.outputCount = outputCount;
		this.activationFunction = activationFunction;

		// Init Values
		this.InitWithRandomValues();
	}

	InitWithRandomValues() {
		for (let inputNode = 0; inputNode < this.inputCount; inputNode++) {
			for (let outputNode = 0; outputNode < this.outputCount; outputNode++) {
				const randomValue = Math.random() * 2 - 1;
				this.weight[inputNode][outputNode] =
					randomValue / Math.sqrt(this.inputCount);
			}
		}
	}

	CalculateOutput(inputValues: number[]) {
		if (inputValues.length != this.weight.length) {
			throw new Error(
				`Invalid Inputs Size :\nExpexted : ${this.weight.length}\nReceived : ${inputValues.length}`,
			);
		}

		let result = new Array<number>(this.biases.length);

		for (let outputIndex = 0; outputIndex < this.outputCount; outputIndex++) {
			let nodeValue = this.biases[outputIndex];

			for (let inputIndex = 0; inputIndex < this.inputCount; inputIndex++) {
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

	ApplyGradients(learnRate: number) {
		for (let outputNode = 0; outputNode < this.outputCount; outputNode++) {
			this.biases[outputNode] -=
				this.costGradientBiases[outputNode] * learnRate;

			for (let inputNode = 0; inputNode < this.inputCount; inputNode++) {
				this.weight[inputNode][outputNode] -=
					this.costGradientWeight[inputNode][outputNode] * learnRate;
			}
		}
	}
}
