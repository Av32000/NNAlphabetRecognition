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
}
