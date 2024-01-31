import { readFileSync, writeFileSync } from 'fs';
import { Dataset, DatasetElement } from './Dataset';
import { Layer } from './Layer';

export type ActivationFunction = 'BinaryStep' | 'Sigmoid' | 'ReLU';

export class NeuralNetwork {
	layers: Layer[] = [];

	constructor(layers?: number[], activationFunction?: ActivationFunction, path?: string) {
		// Setup Layers
		if (layers && activationFunction) {
			for (let i = 0; i < layers.length; i++) {
				const element = layers[i];
				if (i > 0) {
					this.layers.push(new Layer(layers[i - 1], layers[i], activationFunction));
				}
			}
		} else if (path) {
			this.LoadModel(path);
		} else {
			throw new Error('Invalid Constructor');
		}
	}
	Learn(dataset: Dataset, learnRate: number) {
		let i = 0
		dataset.elements.forEach(e => {
			console.log(i);
			i++
			
			this.UpdateAllGradients(e);
		});

		this.layers.forEach(l => {
			l.ApplyGradients(learnRate / dataset.elements.length);
			l.ClearGradients();
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

	UpdateAllGradients(datasetElement: DatasetElement) {
		this.CalculateOutputs(datasetElement.inputs);

		const outputLayer = this.layers[this.layers.length - 1];
		let nodeValues = outputLayer.CalculateOutputLayerNodeValues(datasetElement.expectedOutputs);
		outputLayer.UpdateGradients(nodeValues);

		for (let layerIndex = this.layers.length - 2; layerIndex >= 0; layerIndex--) {
			const hiddenLayer = this.layers[layerIndex];
			nodeValues = hiddenLayer.CalculateHiddenLayerNodeValues(
				this.layers[layerIndex + 1],
				nodeValues,
			);
			hiddenLayer.UpdateGradients(nodeValues);
		}
	}

	ExportModel(path: string) {
		let data: {
			layers: {
				weight: number[][];
				biases: number[];
				activation: ActivationFunction;
				inputCount: number;
				outputCount: number;
			}[];
			struct: number[];
		} = { layers: [], struct: [] };

		data.struct.push(this.layers[0].inputCount);

		this.layers.forEach(layer => {
			data.layers.push({
				weight: layer.weights,
				biases: layer.biases,
				activation: layer.activationFunction,
				inputCount: layer.inputCount,
				outputCount: layer.outputCount,
			});

			data.struct.push(layer.outputCount);
		});

		writeFileSync(path, JSON.stringify(data));
	}

	LoadModel(path: string) {
		const data = JSON.parse(readFileSync(path).toString());
		if (data.layers && data.struct) {
			data.layers.forEach(
				(l: {
					weight: number[][];
					biases: number[];
					activation: ActivationFunction;
					inputCount: number;
					outputCount: number;
				}) => {
					const newLayer = new Layer(l.inputCount, l.outputCount, l.activation);
					newLayer.weights = l.weight;
					newLayer.biases = l.biases;
					this.layers.push(newLayer);
				},
			);
		}
	}
}
