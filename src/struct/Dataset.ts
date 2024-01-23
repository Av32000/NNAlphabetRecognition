export type DatasetElement = {
	inputs: number[];
	expectedOutputs: number[];
};

export class Dataset {
	elements: DatasetElement[] = [];

	AddElement(inputs: number[], expectedOutputs: number[]) {
		this.elements.push({ inputs, expectedOutputs });
	}
}
