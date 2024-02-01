import { readFileSync, writeFileSync } from 'fs';

export type DatasetElement = {
	inputs: number[];
	expectedOutputs: number[];
};

export class Dataset {
	elements: DatasetElement[] = [];
	constructor(path?: string) {
		if (path) {
			this.LoadDataset(path);
		}
	}

	AddElement(inputs: number[], expectedOutputs: number[]) {
		this.elements.push({ inputs, expectedOutputs });
	}

	ExportDataset(path: string) {
		writeFileSync(path, JSON.stringify(this.elements));
	}

	LoadDataset(path: string) {
		this.elements = JSON.parse(readFileSync(path).toString());
	}

	// Utils
	AddPertubations(elem: DatasetElement, percentage: number = 10) {
		const inputs = [...elem.inputs];
		const changesCount = Math.round((inputs.length * percentage) / 100);

		for (let i = 0; i < changesCount; i++) {
			const index = Math.floor(Math.random() * inputs.length);
			inputs[index] = Number(!Boolean(inputs[index]));
		}

		return { inputs, expectedOutputs: elem.expectedOutputs };
	}

	// https://stackoverflow.com/a/2450976/18031156
	Shuffle() {
		let currentIndex = this.elements.length,
			randomIndex;

		while (currentIndex > 0) {
			randomIndex = Math.floor(Math.random() * currentIndex);
			currentIndex--;

			[this.elements[currentIndex], this.elements[randomIndex]] = [
				this.elements[randomIndex],
				this.elements[currentIndex],
			];
		}

		return this;
	}

	Slice(newDatasetCount: number, random = false) {
		if (this.elements.length < newDatasetCount) throw new Error('Invalid Length');

		const newDataset = new Dataset();
		const alreadyUsedIndex: number[] = [];

		for (let i = 0; i < newDatasetCount; i++) {
			let element: DatasetElement;

			if (random) {
				let randomIndex = -1;
				while (alreadyUsedIndex.includes(randomIndex) || randomIndex == -1) {
					randomIndex = Math.floor(Math.random() * this.elements.length);
				}
				element = this.elements[randomIndex];
				alreadyUsedIndex.push(randomIndex);
			} else {
				element = this.elements[i];
			}

			newDataset.AddElement(element.inputs, element.expectedOutputs);
		}

		return newDataset;
	}
}
