import {
	close,
	createReadStream,
	createWriteStream,
	readFileSync,
	readdirSync,
} from 'fs';
import { Dataset } from './struct/Dataset';
import { NeuralNetwork } from './struct/NeuralNetwork';

const learnRate = 0.2;

// const network = new NeuralNetwork([2, 5, 7, 2], 'Sigmoid');

// const dataset = new Dataset();
// for (let i = 0; i < 100; i++) {
// 	const x = Math.random();
// 	const y = Math.random();

// 	const expectedOutputs = x / y > 0.5 ? [0, 1] : [1, 0];

// 	dataset.AddElement([x, y], expectedOutputs);
// }

// for (let i = 0; i < 1000; i++) {
// 	network.Learn(dataset, learnRate);
// }

// console.log(network.DatasetCost(dataset));

// network.ExportModel('model.json');
// dataset.ExportDataset('data.json');

// const network = new NeuralNetwork(undefined, undefined, 'model.json');
// network.ExportModel('model2.json');
// const dataset = new Dataset('data.json');
// dataset.ExportDataset('data2.json');

import { PNG } from 'pngjs';

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
							const idx = (this.width * y + x) << 2; // Décalage pour accéder aux valeurs RGBA
							const r = this.data[idx];
							const g = this.data[idx + 1];
							const b = this.data[idx + 2];

							inputs.push(r + g + b > 600 ? 0 : 1);
						}
					}

					dataset.AddElement(inputs, expected);

					// Fermer le flux de lecture après avoir traité tous les pixels
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

	// Vérifier si la lettre est une lettre majuscule
	const upperCaseLetter = letter.toUpperCase();

	// Obtenir la position de la lettre dans l'alphabet (A=0, B=1, ..., Z=25)
	const position = upperCaseLetter.charCodeAt(0) - 'A'.charCodeAt(0);

	// Assigner la position dans le tableau en ajoutant 1
	if (position >= 0 && position < 36) {
		result[position] = 1; // Commence par 1 pour la lettre A
	}

	return result;
}
CreateDataset();
