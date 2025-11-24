# Kubernetes Deployment Guide

This directory contains Kubernetes manifests for deploying the Health Risk Prediction MLOps system to a Kubernetes cluster.

## Prerequisites

- Kubernetes cluster (v1.20+)
- kubectl configured to access your cluster
- Docker image `health-risk-prediction:latest` built and pushed to a registry accessible by your cluster

## Deployment Steps

### 1. Create Namespace

```bash
kubectl apply -f namespace.yaml
```

### 2. Create Persistent Volume Claims

```bash
kubectl apply -f persistent-volumes.yaml
```

### 3. Create ConfigMap

```bash
kubectl apply -f configmap.yaml
```

### 4. Deploy Services

Deploy services in order:

```bash
# MLflow tracking server
kubectl apply -f mlflow-deployment.yaml

# Federated learning server
kubectl apply -f federated-server-deployment.yaml

# Dashboard
kubectl apply -f dashboard-deployment.yaml
```

### 5. Run Training Job (Optional)

```bash
kubectl apply -f training-job.yaml
```

## Accessing Services

### Dashboard

The dashboard service is exposed as a LoadBalancer. Get the external IP:

```bash
kubectl get service dashboard-service
```

Then access at: `http://<EXTERNAL_IP>`

### MLflow UI

Port-forward to access MLflow:

```bash
kubectl port-forward service/mlflow-service 5000:5000 -n health-risk-prediction
```

Then access at: `http://localhost:5000`

### Federated Server

The federated server is accessible within the cluster at:
- `federated-server-service:8080`

## Scaling

### Scale Dashboard

```bash
kubectl scale deployment dashboard --replicas=3 -n health-risk-prediction
```

## Monitoring

### Check Pod Status

```bash
kubectl get pods -n health-risk-prediction
```

### Check Logs

```bash
# Dashboard logs
kubectl logs -f deployment/dashboard -n health-risk-prediction

# MLflow logs
kubectl logs -f deployment/mlflow-server -n health-risk-prediction

# Federated server logs
kubectl logs -f deployment/federated-server -n health-risk-prediction
```

## Cleanup

To remove all resources:

```bash
kubectl delete namespace health-risk-prediction
```

## Production Considerations

1. **Image Registry**: Update image references to use your container registry
2. **Storage**: Configure appropriate storage classes for your cluster
3. **Secrets**: Add secrets for API keys, database credentials, etc.
4. **Ingress**: Configure Ingress controller for external access
5. **Resource Limits**: Adjust resource requests/limits based on your cluster capacity
6. **High Availability**: Consider using StatefulSets for stateful services
7. **Monitoring**: Integrate with Prometheus/Grafana for monitoring
8. **Backup**: Set up backups for persistent volumes

## Troubleshooting

### Pods Not Starting

```bash
# Check pod events
kubectl describe pod <pod-name> -n health-risk-prediction

# Check pod logs
kubectl logs <pod-name> -n health-risk-prediction
```

### Storage Issues

```bash
# Check PVC status
kubectl get pvc -n health-risk-prediction

# Check PV status
kubectl get pv
```

### Network Issues

```bash
# Check services
kubectl get svc -n health-risk-prediction

# Test connectivity
kubectl run -it --rm debug --image=busybox --restart=Never -- ping mlflow-service
```

