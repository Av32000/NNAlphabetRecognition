import { createReadStream, readdirSync } from 'fs';
import { Dataset } from './struct/Dataset';
import { NeuralNetwork } from './struct/NeuralNetwork';
import { PNG } from 'pngjs';

const learnRate = 0.2;

async function CreateDataset() {
	const dataset = new Dataset();

	const files = readdirSync('data/test');
	console.log(files);
	let i = 0;
	for (const f of files) {
		console.log(i);

		await new Promise(resolve => {
			const stream = createReadStream('data/test/' + f)
				.pipe(
					new PNG({
						filterType: 4,
					}),
				)
				.on('parsed', function () {
					const inputs = [];
					const expected = generateArray(f[0]);
					for (let y = 0; y < this.height; y++) {
						for (let x = 0; x < this.width; x++) {
							const idx = (this.width * y + x) << 2;
							const r = this.data[idx];
							const g = this.data[idx + 1];
							const b = this.data[idx + 2];

							inputs.push(r + g + b > 600 ? 0 : 1);
						}
					}

					dataset.AddElement(inputs, expected);
					stream.end();
					i++;
					resolve(0);
				});
		});
	}
	dataset.ExportDataset('test.json');
}

function generateArray(letter: string): number[] {
	const result: number[] = Array(26).fill(0);
	const upperCaseLetter = letter.toUpperCase();
	const position = upperCaseLetter.charCodeAt(0) - 'A'.charCodeAt(0);
	if (position >= 0 && position < 36) {
		result[position] = 1;
	}

	return result;
}

function TranslateResult(outputs: number[]) {
	let maxIndex = 0;

	for (let i = 0; i < outputs.length; i++) {
		if (outputs[i] > outputs[maxIndex]) {
			maxIndex = i;
		}
	}

	return String.fromCharCode(65 + maxIndex);
}

const testDataset = new Dataset('data/test.json');
const trainDataset = new Dataset('data/train.json');

const network = new NeuralNetwork([784, 100, 26], 'Sigmoid');
network.SetupStats(TranslateResult);

// const sliced = trainDataset.Shuffle().Slice(200);
trainDataset.Shuffle();
console.time('Test');
for (let i = 0; i < 50; i++) {
	network.Learn(trainDataset, learnRate, 4);
	console.log(network.correctPercentages[network.correctPercentages.length - 1]);

	if (i % 10 == 0) {
		network.ExportModel('data/model.json');
		console.log('Model Saved !');
	}
}
network.ExportModel('data/model.json');
console.log('Model Saved !');

console.timeEnd('Test');

function writeFileSync(cheminFichier: string, imageBuffer: any) {
	throw new Error('Function not implemented.');
}
// CreateDataset();
