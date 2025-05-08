kubectl apply -f ray/simple_ray_cluster.yaml
kubectl expose svc my-ray-cluster-head-svc --type=NodePort --target-port=8265 --name=ray-cluster-dashboard-nodeport-svc
kubectl expose svc my-ray-cluster-head-svc --type=NodePort --target-port=10001 --name=ray-cluster-client-nodeport-svc