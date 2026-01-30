# Structure- and Reliability-aware Knowledge Graph Enhancement for Drug–Drug Interaction Prediction


# Installation & Dependencies


SR-KGE is mainly tested in a Linux environment, and its dependencies are listed below.


| Package         | Version  |
|-----------------|----------|
| python          | 3.9.21   |
| rdkit           | 2024.3.6 |
| pytorch         | 2.5.0    |
| cuda            | 12.4     |
| torch-geometric | 2.0.2    |
| torch-scatter   | 2.1.2    |
| torch-sparse    | 0.6.18   |
| torchvision     | 0.20.0   |
| scikit-learn    | 1.5.1    |
| tqdm            | 4.67.1   |
| networkx        | 3.2.1    |
| matplotlib      | 3.9.4    |
| pandas          | 2.2.3    |
| numpy           | 1.26.4   |


# Run SR-KGE

SR-KGE runs in two stages: (1) KG completion to generate an enhanced KG, and (2) DDI prediction using the enhanced KG. 

You can train SR-KGE with the following command:

## Step 1: Generate the enhanced KG


```bash
cd kg_completion
python kg_completion_output.py
```

Enhanced KG is saved in `kg_completion/kg_completion_output/drugbank/kg` and should be moved to `datasets/drugbank`

## Step 2: Run DDI prediction


```bash
python main.py
```


or


```bash
python main.py --dataset drugbank --extractor probability
```




