# Docker Swarm Cluster:
To create our docker swarm we need to follow this tutorial: https://docs.docker.com/engine/swarm/swarm-tutorial/create-swarm/

## Configure NTP in all nodes:

To configure NTP follow tutorial on the link: https://ntp.br/guia/linux/.

After configure node, we need to restart docker using:

```
sudo systemctl restart docker
```

# Metrics:
We will use in our cluster Traefik as our reverse proxy, Prometheus to collect our metrics and Grafana to visualize these metrics.

### Traefik
Compose and config files to traefik are in the folder /traefik.
To run this compose file we need to create the network used by traefik and other services.

```
docker network create --driver=overlay --attachable traefik-public
```

Inside traefik folder, we need to run command below.

```
docker stack deploy -c docker-compose.yml cl_traefik
```

This command will deploy traefik as a stack named cl_traefik.
If it is necessary we can run the command below to remove this stack.

```
docker stack rm cl_traefik
```

### Grafana
Compose and config files to grafana are in folder /grafana.

Before we start the stack of grafana, we need to create a volume for grafana data.

```
docker volume create --driver local grafana_data
```

Inside grafana folder, we need to run command below.

```
docker stack deploy -c docker-compose.yml cl_grafana
```

This command will deploy grafana as a stack named cl_grafana.
If it is necessary we can run the command below to remove this stack.

```
docker stack rm cl_grafana
```

### Prometheus
Compose and config files to prometheus are in folder /prometheus.

```
docker volume create --driver local prometheus_data
```

It is necessary run the command below to retrieve some information about nodes. This will create node_meta

```
docker config create node_exporter_entrypoint config/node_exporter_entrypoint
```

Inside prometheus folder, we need to run the command below.

```
docker stack deploy -c docker-compose.yml cl_prometheus
```

This command will deploy prometheus, node-exporter and cadvisor as a stack named cl_prometheus.
If it is necessary we can run the command below to remove this stack.

```
docker stack rm cl_prometheus
```

# Cassandra Cluster
Compose and config files to cassandra cluster are in folder /cassandra.

First we need to create volumes to our nodes.

```
docker volume create --driver local cassandra1_data
```

```
docker volume create --driver local cassandra2_data
```

```
docker volume create --driver local cassandra3_data
```

```
docker volume create --driver local cassandra4_data
```

```
docker volume create --driver local cassandra5_data
```

It is necessary to create the following config

```
docker config create wait-for-seed config/wait-for-seed.sh
```


Inside cassandra folder, we need to run the command below.

```
docker stack deploy -c docker-compose.yml cl_cassandra
```

This command will deploy all cassandra nodes as a stack named cl_cassandra.
If it is necessary we can run the command below to remove this stack.

```
docker stack rm cl_cassandra
```