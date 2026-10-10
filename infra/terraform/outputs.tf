output "public_ip" {
  value = oci_core_instance.node.public_ip
}

output "next_steps" {
  value = <<-EOT
    ssh ubuntu@${oci_core_instance.node.public_ip} 'sudo cat /etc/rancher/k3s/k3s.yaml' > kubeconfig-oci
    sed -i 's/127.0.0.1/${oci_core_instance.node.public_ip}/' kubeconfig-oci
    KUBECONFIG=kubeconfig-oci helm upgrade --install powerfc ../../deploy/helm/powerfc \
      --namespace powerfc --create-namespace \
      --set image.repository=ghcr.io/<owner>/powerfc-app,image.tag=<sha> \
      --set mlflowImage.repository=ghcr.io/<owner>/powerfc-mlflow,mlflowImage.tag=<sha>
    open http://${oci_core_instance.node.public_ip}/docs
  EOT
}
