import { NeuralNetwork } from './struct/NeuralNetwork';

const network = new NeuralNetwork([2, 5, 7, 3], 'Sigmoid');

console.log(network.CalculateOutputs([0.2, 0.5]));
