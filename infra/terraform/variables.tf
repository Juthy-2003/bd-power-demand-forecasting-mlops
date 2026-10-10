variable "tenancy_ocid" { type = string }
variable "user_ocid" { type = string }
variable "fingerprint" { type = string }
variable "private_key_path" { type = string }
variable "region" {
  type    = string
  default = "ap-mumbai-1" # closest OCI region to Dhaka; check Always Free A1 capacity there
}
variable "compartment_ocid" { type = string }

variable "ssh_public_key_path" {
  type    = string
  default = "~/.ssh/id_ed25519.pub"
}

variable "admin_cidr" {
  description = "Your own public IP as a /32: only this address may SSH or reach the Kubernetes API."
  type        = string
}

# Sized to the Always Free limit reported for 2026 (2 OCPU / 12 GB). That
# figure comes from a secondary source; check Oracle's Always Free page
# before applying, because exceeding the free allowance is billed.
variable "ocpus" {
  type    = number
  default = 2
}
variable "memory_gb" {
  type    = number
  default = 12
}
variable "boot_volume_gb" {
  type    = number
  default = 50
}

variable "k3s_version" {
  type    = string
  default = "v1.33.4+k3s1"
}
