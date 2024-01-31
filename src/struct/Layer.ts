import { ActivationFunction } from './NeuralNetwork';

export class Layer {
	weights: number[][] = [];
	biases: number[] = [];
	activationFunction: ActivationFunction;

	inputCount: number;
	outputCount: number;

	costGradientWeight: number[][] = [];
	costGradientBiases: number[] = [];

	weightedInputs: number[] = [];
	activations: number[] = [];
	inputs: number[] = [];
	constructor(inputCount: number, outputCount: number, activationFunction: ActivationFunction) {
		// Setup weight
		this.weights = new Array<number[]>(inputCount);
		for (let i = 0; i < inputCount; i++) {
			this.weights[i] = new Array<number>(outputCount).fill(0);
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
				this.weights[inputNode][outputNode] = randomValue / Math.sqrt(this.inputCount);
			}
		}
	}

	CalculateOutput(inputValues: number[]) {
		if (inputValues.length != this.weights.length) {
			throw new Error(
				`Invalid Inputs Size :\nExpexted : ${this.weights.length}\nReceived : ${inputValues.length}`,
			);
		}

		let result = new Array<number>(this.biases.length);
		this.weightedInputs = [];
		this.activations = [];
		this.inputs = inputValues;

		for (let outputIndex = 0; outputIndex < this.outputCount; outputIndex++) {
			let nodeValue = this.biases[outputIndex];

			for (let inputIndex = 0; inputIndex < this.inputCount; inputIndex++) {
				const inputValue = inputValues[inputIndex];
				nodeValue += inputValue * this.weights[inputIndex][outputIndex];
			}

			this.weightedInputs.push(nodeValue);

			const activatedOutput = this.ActivateOutput(nodeValue);

			this.activations.push(activatedOutput);
			result[outputIndex] = activatedOutput;
		}

		return result;
	}

	CalculateOutputLayerNodeValues(expectedOutputs: number[]) {
		let nodeValues = new Array<number>(expectedOutputs.length);

		for (let i = 0; i < nodeValues.length; i++) {
			const costDerivative = this.NodeCostDerivate(this.activations[i], expectedOutputs[i]);
			const activationDerivative = this.ActivateOutputDerivate(this.weightedInputs[i]);

			nodeValues[i] = activationDerivative * costDerivative;
		}

		return nodeValues;
	}

	CalculateHiddenLayerNodeValues(oldLayer: Layer, oldNodeValues: number[]) {
		const newNodeValues = new Array<number>(this.outputCount);

		for (let newNodeIndex = 0; newNodeIndex < newNodeValues.length; newNodeIndex++) {
			let newNodeValue = 0;
			for (let oldNodeIndex = 0; oldNodeIndex < oldNodeValues.length; oldNodeIndex++) {
				const weightedInputDerivative = oldLayer.weights[newNodeIndex][oldNodeIndex];
				newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex];
			}
			newNodeValue *= this.ActivateOutputDerivate(this.weightedInputs[newNodeIndex]);
			newNodeValues[newNodeIndex] = newNodeValue;
		}

		return newNodeValues;
	}

	ActivateOutput(output: number) {
		switch (this.activationFunction) {
			case 'BinaryStep':
				return output < 0 ? 0 : 1;
			case 'Sigmoid':
				return 1 / (1 + Math.exp(-output));
			case 'ReLU':
				return (Math.exp(output) - Math.exp(-output)) / (Math.exp(output) + Math.exp(-output));
		}
	}

	ActivateOutputDerivate(output: number) {
		switch (this.activationFunction) {
			case 'BinaryStep':
				throw new Error('Not Implemented');
			case 'Sigmoid':
				const activation = this.ActivateOutput(output);
				return activation * (1 - activation);
			case 'ReLU':
				throw new Error('Not Implemented');
		}
	}

	NodeCost(output: number, expectedOutput: number) {
		const error = output - expectedOutput;
		return error * error;
	}

	NodeCostDerivate(output: number, expectedOutput: number) {
		return 2 * (output - expectedOutput);
	}

	UpdateGradients(nodeValues: number[]) {
		for (let outputIndex = 0; outputIndex < this.outputCount; outputIndex++) {
			for (let inputIndex = 0; inputIndex < this.inputCount; inputIndex++) {
				const derivativeCostWeight = this.inputs[inputIndex] * nodeValues[outputIndex];

				this.costGradientWeight[inputIndex][outputIndex] += derivativeCostWeight;
			}

			const derivativeCostBias = 1 * nodeValues[outputIndex];
			this.costGradientBiases[outputIndex] += derivativeCostBias;
		}
	}

	ApplyGradients(learnRate: number) {
		for (let outputNode = 0; outputNode < this.outputCount; outputNode++) {
			this.biases[outputNode] -= this.costGradientBiases[outputNode] * learnRate;

			for (let inputNode = 0; inputNode < this.inputCount; inputNode++) {
				this.weights[inputNode][outputNode] -=
					this.costGradientWeight[inputNode][outputNode] * learnRate;
			}
		}
	}

	ClearGradients() {
		this.costGradientBiases = [];
		this.costGradientWeight = [];

		this.weightedInputs = [];
		this.activations = [];
		this.inputs = [];
	}
}
