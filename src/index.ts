import { createReadStream, readdirSync } from 'fs';
import { PNG } from 'pngjs';
import { Dataset } from './struct/Dataset';
import { NeuralNetwork } from './struct/NeuralNetwork';

// Settings
const learnRate = 0.24;
const perturbationsPercentage = 4;
const batchSize = 75;

async function CreateDataset(inputPath: string, resultPath: string) {
	const dataset = new Dataset();

	const files = readdirSync(inputPath);
	let i = 0;
	for (const f of files) {
		console.log(i);
		await new Promise(resolve => {
			const stream = createReadStream(inputPath + '/' + f)
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
	dataset.ExportDataset(resultPath);
}

function parseImage(path: string): Promise<number[]> {
	return new Promise((resolve, reject) => {
		const inputs: number[] = [];

		createReadStream(path)
			.pipe(new PNG({ filterType: 4 }))
			.on('parsed', function () {
				for (let y = 0; y < this.height; y++) {
					for (let x = 0; x < this.width; x++) {
						const idx = (this.width * y + x) << 2;
						const r = this.data[idx];
						const g = this.data[idx + 1];
						const b = this.data[idx + 2];

						inputs.push(r + g + b > 600 ? 0 : 1);
					}
				}
				resolve(inputs);
			})
			.on('error', error => {
				reject(error);
			});
	});
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

async function TrainNetwork(
	trainPath: string,
	testPath: string,
	network: NeuralNetwork,
	resultPath: string,
	iterationsCount: number,
) {
	network.SetupStats(TranslateResult);
	const testDataset = new Dataset(testPath);
	const trainDataset = new Dataset(trainPath);

	trainDataset.Shuffle();

	console.log(
		`Training Settings :\n- Iterations Count : ${iterationsCount}\n- Learn Rate : ${learnRate}\n- Perturbations Percentage : ${perturbationsPercentage}\n- Batch Size : ${batchSize}`,
	);

	console.time('Train');
	for (let i = 0; i < iterationsCount; i++) {
		network.Learn(trainDataset, learnRate, perturbationsPercentage, batchSize);
		console.log(
			`[Iteration ${i + 1}/${iterationsCount}] ${
				network.correctPercentages[network.correctPercentages.length - 1]
			}%`,
		);

		if (i % 10 == 0 && i != 0) {
			network.ExportModel(resultPath);
			console.log('Model Saved !');
		}
	}
	network.ExportModel(resultPath);
	console.log('Model Saved !');

	console.timeEnd('Train');

	console.log('Train : ' + network.TestDataset(trainDataset, TranslateResult));
	console.log('Test : ' + network.TestDataset(testDataset, TranslateResult));
}

async function RunOnImage(path: string, network: NeuralNetwork) {
	parseImage(path).then(inputs => {
		const outputs = network.CalculateOutputs(inputs);
		const finalArray = outputs.map((nombre, index) => ({
			letter: String.fromCharCode(65 + index),
			value: nombre,
			percentage:
				(nombre / outputs.reduce((acc, nombre) => acc + nombre, 0)) * 100,
		}));

		finalArray.sort((a, b) => b.value - a.value);

		finalArray.forEach(element => {
			console.log(`${element.letter} => ${element.percentage.toFixed(2)}%`);
		});
	});
}
