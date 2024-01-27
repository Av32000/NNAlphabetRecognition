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

// const network = new NeuralNetwork(undefined, undefined, 'model.json');
// network.ExportModel('model2.json');
