# Smartness Cassandra Stress RS

This repository contains code to conduct experiments with Regression Models, Non-IID analysis and Federated Learning.

### Regression Models:

Experiments with a Wide and Deep neural network to conduct our regression task. Besides this model, this repository 
contains some notebooks with random forest, xgboost and other algorithms.

### Non-IID analysis:

Some analysis was conducted using our datasets to check if our datasets were Non-IID datasets.

### Federated Learning:

Federated Learning was developed using Flower framework. The code for Federated Learning is on /smartness-flwr folder.
Experiments was conduct using horizontal architecture.

**Paper:** "An Experimental Framework for Studying Non-IID Data in Federated Learning for Network Telemetry"
**Abstract:**
_The increasing complexity of emerging 5G and 6G network environ-
ments has intensified the need for data-driven automation under heterogeneous
and dynamic conditions. Federated Learning (FL) is a promising paradigm in
this context. This paper presents an experimental framework to generate real-
istic Non-Independent and Identically Distributed (Non-IID) datasets through
controlled execution of a distributed service and telemetry collection, aiming to
improve the applicability of FL in network automation. Using Apache Cassan-
dra as a representative cloud-native application, we construct datasets exhibit-
ing temporal and structural heterogeneity. We statistically characterize these
datasets and evaluate their impact on regression models and horizontal fed-
erated learning using a Wide & Deep architecture. Results show that while
horizontal federation improves generalization compared to direct cross-dataset
transfer, its performance degrades under pronounced structural Non-IID condi-
tions, highlighting both its potential and limitations._

Two other repositories make up the entire environment for the experiments.

[Smartness Cassandra Stress RS](https://github.com/EricssonResearch/smartness-cassandra-stress-rs), contains code to our client and python scripts for our load generator and other utilities.

[Datasets](https://github.com/EricssonResearch/telemetry_sbrc_2026), contains dataset produced in our experiments.

# Structure

This repository is structured as follows:

### docker_swarm_cassandra:
Contains compose files to start a Docker Swarm used in our experiments. [README.md](docker_swarm_cassandra/README.md) contains instructions to start this environment.

#### System requirement:
Seven VMs are running Ubuntu 24.04 LTS: one with 300 GB of storage (to store metrics collected by Prometheus) and six with 80 GB. Each features 4 vCPUs and 16 GB of RAM.

Six of these VMs are used in our Docker Swarm cluster. The VM with 300 GB serves as the manager node, hosting Traefik, Grafana, and Prometheus. An Apache Cassandra node is installed on each of the remaining five VMs.

We installed ```node_exporter``` and ```cAdvisor``` on all Docker Swarm nodes to export metrics from both the hosts and the containers.

### images:
Contains images generated  by our notebook scripts.

### nn_regression_analysis:
Contains scripts used in our experiments with various neural networks.

### smartness-flwr:
Contains code used in our federated learning setup. The Flower framework was used to federate our datasets.

### .ipynb
Contains notebooks used to conduct analysis on our datasets.

# Badges considered

The authors consider the following badges as part of the evaluation process:

- Artifacts Available (SeloD)
- Artifacts Functional (SeloF)

# Basic information

## Python scripts

- Python 3.12.3

We used [asdf](https://asdf-vm.com/) to manage our Python versions. The .tool-versions file in the project folder already specifies the exact version required for this project.

## System requirements

- Ubuntu 24.04
- 16Gb RAM
- 13th Gen Intel® Core™ i7-13800H × 20
- 250 Gb Disk

# Dependencies

To run our Python scripts it is necessary to install Python 3.12.3.

The commands below need to be ran inside project's folder.

It is necessary to create a virtual environment by running the command below:

```
python -m venv .venv
```

The command above will create a folder .venv, it is necessary to activate the virtual environment using command below:

```
source .venv/bin/activate
```

After starting the virtual environment, you can install all dependencies using the command below:

```
pip install -r requirements.txt
```

# Minimal test

The commands below need to be ran inside project's folder. First it is necessary to create two folder to store datasets and models respectively.

```
mkdir dataset
mkdir models
```

Then, it necessary to download datasets available on [Datasets](https://github.com/EricssonResearch/telemetry_sbrc_2026) and store in folder dataset. After download, it is necessary to unzip datasets using commands below:

```
cd dataset

unzip t100_X_y.zip -d t100
unzip t300_X_y.zip -d t300
unzip t500_X_y.zip -d t500

cd ..
```


### Regression analysis:

In our experiment we use the script _nn_regression_analysis/fed_mlp_regression_analysis.py_ to conduct our training and evaluation using a deep model.

Below there is the command to run training step:

```
python nn_regression_analysis/fed_mlp_regression_analysis.py --data dataset/t100/prometheus_metrics_wide.csv --target_data dataset/t100/t100_write.csv --target_col w_95th_percentile --model_path models/t100_write_w95perc.pth --rounds -1
```

To evaluate, use the command below:

```
python smartness-flwr/smartness_flwr/evaluate_global_model.py --data dataset/t100/prometheus_metrics_wide.csv --target dataset/t100/t100_write.csv --col w_95th_percentile --rem-perc 95 --model-path models/t100_write_w95perc.pth --rounds -1
```

### Federated experiments:

The commands below will run our federated experiments with 3 clients. To change the number of clients, it is necessary to modify the script _smartness-flwr/smartness_flwr/server_mlp.py_ (line 17).


Server:

```
python smartness-flwr/smartness_flwr/server_mlp.py --sample-data dataset/t100/prometheus_metrics_wide.csv --sample-target dataset/t100/t100_write.csv --col w_95th_percentile --rem-perc 95 --model-path models/global_model_3c_w_w95_zc95.pth
```

Client:

```
python smartness-flwr/smartness_flwr/client_mlp.py --cid 0 --server 127.0.0.1:8080 --data dataset/t100/prometheus_metrics_wide.csv --target dataset/t100/t100_write.csv --col w_95th_percentile --rem-perc 95

python smartness-flwr/smartness_flwr/client_mlp.py --cid 1 --server 127.0.0.1:8080 --data dataset/t300/prometheus_metrics_wide.csv --target dataset/t300/t300_write.csv --col w_95th_percentile --rem-perc 95

python smartness-flwr/smartness_flwr/client_mlp.py --cid 2 --server 127.0.0.1:8080 --data dataset/t500/prometheus_metrics_wide.csv --target dataset/t500/t500_write.csv --col w_95th_percentile --rem-perc 95
```

Evalute global model:

```
python smartness-flwr/smartness_flwr/evaluate_global_model.py --data dataset/t300/prometheus_metrics_wide.csv --target dataset/t300/t300_write.csv --col w_95th_percentile --rem-perc 95 --model-path models/global_model_3c_w_w95_zc95.pth --rounds -1
```

# LICENSE

This software is under MIT-License. For more information please read LICENSE file.