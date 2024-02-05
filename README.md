# Alphabet Recognition - Neural Network

Experiment to create an artificial neural network in Typescript to recognize capital letters.

Inspired by a video by [Sebastian Lague](https://www.youtube.com/@SebastianLague): [How to Create a Neural Network (and Train it to Identify Doodles)](https://www.youtube.com/watch?v=hfMk-kjRv4c)

## Installation

1. Clone Repo
2. Install Dependencies (`npm install`)
3. Create new folder : `data`
4. Put dataset files with this scheme :
   - `data/train/[letter]\_[index].png`
   - `data/test/[letter]\_[index].png`
5. Run `CreateDatase()` to create `dataset.json`

## Training Test

The first training session used Huggins Face's dataset : [pittawat/letter_recognition](https://huggingface.co/datasets/pittawat/letter_recognition)

Params :

- Layers : [784, 100, 26]
- LearnRate : 0.24
- Activation Function : ReLU
- Perturbations : 4%

Results (816 iterations) :

- TrainDataset : 93%
- TestDataset 91%
