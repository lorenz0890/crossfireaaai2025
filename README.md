# Crossfire: An Elastic Defense Framework for Graph Neural Networks under Bit Flip Attacks

## Installation
1) Crossfire is seamlessly integrated in the PyQ quantization package. Hence for installation, navigate to crossfire/pyq and follow the steps documented in README.md to install PyQ.

2) After installing PyQ, new quantized models can be trained by running any of the scripts found in ibfa/pyq/graph. Make sure to adapt the paths in the script to your local environment. 
For a quick start, pre-trained quantized models can be downloaded [here](https://ucloud.univie.ac.at/index.php/s/aY5e3b6Jdyy5HTa).

3) Put the quantized models produced in the previous step either in crossfire/models or navigate to crossfire/pyq/run_bfa_{model}_{dataset}.py and adapt the paths to your local environment. They should in point to the locations where the quantized models produced in step 2. are stored.

## Execution 
From the root directory of the repository (crossfire), run
``python experiments_crossfire.py --type PBFA --data ogbg-molhiv --n 1 --k 25 --sz 32 --lim 0.1 --npc 0.1 --npg 1.3``
to execute PBFA attack on Crossfire defended GIN (or another model for which you put the path in the python script) on dataset ogbg-molhiv with k=25 BFA attack runs, batch size sz=32, 1 percent honeypots and base gamma of 1.3 and return results as average of n=1 repetitions of the experiment. 

Example output:

PBFA ogbg-molhiv 1 25 32 0.1 1.33
Start time 16/12/2024, 17:13:27
Downloading http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/hiv.zip
Downloaded 0.00 GB: 100%|███████████████████████████████████████████████████████████████████████████| 3/3 [00:03<00:00,  1.20s/it]
Extracting dataset/hiv.zip
Processing...
Loading necessary files...
This might take a while.
Processing graphs...
100%|███████████████████████████████████████████████████████████████████████████████████| 41127/41127 [00:00<00:00, 103060.98it/s]
Converting graphs into PyG objects...
100%|████████████████████████████████████████████████████████████████████████████████████| 41127/41127 [00:00<00:00, 44356.27it/s]
Saving...
Done!
params 1885241
PBFA data found
Iteration: 100%|████████████████████████████████████████████████████████████████████████████████| 129/129 [00:02<00:00, 46.80it/s]
Defensive gradient ranking
Defensive gradient ranking
Defensive gradient ranking
Defensive gradient ranking
Defensive gradient ranking
Defensive gradient ranking
Defensive gradient ranking
Defensive gradient ranking
Defensive gradient ranking
Defensive gradient ranking
Ranking done
Iteration: 100%|████████████████████████████████████████████████████████████████████████████████| 129/129 [00:02<00:00, 43.61it/s]
{'rocauc': 0.7063268892794377} {'rocauc': 0.7071534792097183}
0 0.20354768633842468 0.20354768633842468
0 0.21088829636573792 0.21088829636573792
0 0.21797499060630798 0.21797499060630798
0 0.22610993683338165 0.22610993683338165
0 0.23520433902740479 0.23520433902740479
0 0.24364016950130463 0.24364016950130463
0 0.2532568573951721 0.2532568573951721
0 0.26333823800086975 0.26333823800086975
0 0.27269142866134644 0.27269142866134644
0 0.2838857173919678 0.2838857173919678
0 0.29443252086639404 0.29443252086639404
0 0.307125449180603 0.307125449180603
0 0.32016223669052124 0.32016223669052124
0 0.33408743143081665 0.33408743143081665
0 0.34726735949516296 0.34726735949516296
0 0.3623252809047699 0.3623252809047699
0 0.3780025839805603 0.3780025839805603
0 0.3927331864833832 0.3927331864833832
0 0.40898841619491577 0.40898841619491577
0 0.42547479271888733 0.42547479271888733
0 0.44326990842819214 0.44326990842819214
0 0.46200528740882874 0.46200528740882874
0 0.48159754276275635 0.48159754276275635
0 0.5006505250930786 0.5006505250930786
0 0.5208374857902527 0.5208374857902527
Iteration: 100%|████████████████████████████████████████████████████████████████████████████████| 129/129 [00:02<00:00, 48.86it/s]
Crossfire Repair
Iteration: 100%|████████████████████████████████████████████████████████████████████████████████| 129/129 [00:02<00:00, 47.28it/s]
Mod Net {'rocauc': 0.7071534792097183} Pre Repair: {'rocauc': 0.6730151219606404} Post Repair: {'rocauc': 0.7071563761370441}
Current time: 16/12/2024, 17:13:55 Completed: 100.0 % Duration per experiment: 28.45 s ETA: 16/12/2024, 17:13:55
Clean GNN 0.71 (0.0), Mod GNN 0.71 (0.0) , Post-BFA GNN 0.67 (0.0), Repaired GNN 0.71 (0.0), Flips 25.0 (0.0), Detection rate 1.0 (0.0), Recovery rate 1.0 (0.0), Attack cost 0.24 (0.01), Failures 0
Start time 16/12/2024, 17:13:55

Our implementation of IBFA is based on the code of the original [Progressive Bitflip Attack](https://github.com/elliothe/Neural_Network_Weight_Attack)
