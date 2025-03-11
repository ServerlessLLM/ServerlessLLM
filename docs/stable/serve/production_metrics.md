---
sidebar_position: 2
---

# Integration with Prometheus
ServerlessLLM supports integration with prometheus. Due to its distributed nature, ServerlessLLM push service level metrics to prometheus pushgateway, and scrapes metrics from the gateway with a prometheus server.

# Example to Report Metrics to Prometheus
1. Start a prometheus pushgateway at any node that is reachable by the controller node and worker nodes, here we simply set it to `localhost`.
prometheus pushgateway download link: https://github.com/prometheus/pushgateway/releases
```
./pushgateway 
```
It will by default listen to port 9091
2. Start prometheus server at the same node
prometheus quickstart guide: https://prometheus.io/docs/prometheus/latest/getting_started/
```
./prometheus --config.file=prometheus.yml
```
Here we provide an example prometheus.yml below:
```yml
# my global config
global:
  scrape_interval: 15s # Set the scrape interval to every 15 seconds. Default is every 1 minute.
  evaluation_interval: 15s # Evaluate rules every 15 seconds. The default is every 1 minute.
  # scrape_timeout is set to the global default (10s).

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
          # - alertmanager:9093

# Load rules once and periodically evaluate them according to the global 'evaluation_interval'.
rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

# A scrape configuration containing exactly one endpoint to scrape:
# Here it's Prometheus itself.
scrape_configs:
  # The job name is added as a label `job=<job_name>` to any timeseries scraped from this config.
  - job_name: "pushgateway"
    honor_labels: true

    # metrics_path defaults to '/metrics'
    # scheme defaults to 'http'.

    static_configs:
      - targets: ["localhost:9091"]
```
3. Start ServerlessLLM
Start ServerlessLLM as indicated in [quickstart](./quickstart.md), when starting sllm-serve, use `--pushgateway_url` to pass pushgateway url:
```
sllm-serve --pushgateway_url localhost:9091
```
Then everything is done, you can query metrics in your prometheus server. You can also visulize it in [grafana](https://grafana.com/docs/grafana/latest/getting-started/get-started-grafana-prometheus/)

