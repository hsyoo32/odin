# ODIN
- The repository is the implementation of [ODIN](https://dl.acm.org/doi/10.1145/3543507.3583271)
Disentangling Degree-related Biases and Interest for Out-of-Distribution Generalized Directed Network Embedding
Hyunsik Yoo, Yeon-Chang Lee, Kijung Shin, Sang-Wook Kim
WWW '23: Proceedings of the ACM Web Conference 2023

## Requirements
- python 37
- pytorch v1.13.0
- scikit-learn
- tqdm
- absl-py
```bash
pip install absl-py
```
- snap (Optional for "_data_preprocessing.py")
```bash
    python -m pip install snap-stanford (https://snap.stanford.edu/snappy/index.html)
```

## Usage

You can use directly the following command to run the odin and save its learned embeddings in "--emb_file".
Warning: you may want to specify the "input_file" and "--emb_file" on your own.
```bash
cd src
python app.py --embedding_size=20 --epochs=200 --option=odin --disen_weight=0.5 --input_file=../_Data/ciaodvd/noniid-in-barrier_1.0/u1/u1.edgelist --emb_file=../_Emb/ciaodvd/noniid-in-barrier_1.0/odin/u1_odin_200_4_odin_0.5_dim60.emb --neg_sample_rate=4
```
- embedding_size: the dimension of the embedding
- epochs: the number of epochs
- option: the option of the algorithm
- disen_weight: the weight of the disentanglement loss
- input_file: the input file
- emb_file: the embedding saving file
- neg_sample_rate: the number of negative samples

(Recommended) You can also use the "tester.py" and "_execute_methods.py" to run the odin with the user-specified hyperparameters, as well as to evaluate the embeddings via the link prediction tasks.
```bash
python tester.py
```
In "_tester.py", you can change the parameters of the ODIN and the datasets.


In "_Data/", all four non-iid-splitted datasets used in the paper are avaliable. If you want to reproduce the dataset from the raw dataset, "_data_preprocessing.py" can be used.
```bash
python _data_preprocessing.py
```
