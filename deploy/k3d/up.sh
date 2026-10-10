#!/usr/bin/env bash
# Local Kubernetes: k3d (k3s in Docker) + the same Helm chart as the cloud node.
# Needs: docker, k3d, helm, kubectl  (brew install k3d helm kubectl)
set -euo pipefail
cd "$(dirname "$0")/../.."
CLUSTER=${CLUSTER:-powerfc}

if ! k3d cluster list "$CLUSTER" >/dev/null 2>&1; then
  # Port 8088 on the host → Traefik ingress (port 80) in the cluster.
  k3d cluster create "$CLUSTER" --agents 0 -p "8088:80@loadbalancer" --wait
fi

docker build -q -t powerfc-app:local .
docker build -q -f Dockerfile.mlflow -t powerfc-mlflow:local .
k3d image import powerfc-app:local powerfc-mlflow:local -c "$CLUSTER"

helm upgrade --install powerfc deploy/helm/powerfc \
  --namespace powerfc --create-namespace \
  --set image.pullPolicy=Never --wait --timeout 15m

kubectl -n powerfc wait --for=condition=complete job -l app=bootstrap --timeout=20m
kubectl -n powerfc get pods
echo "Forecast API: http://localhost:8088/docs"
