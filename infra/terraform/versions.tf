terraform {
  required_version = ">= 1.6"
  required_providers {
    oci = {
      source  = "oracle/oci"
      version = ">= 6.0"
    }
  }
  # State holds resource IDs and must never be committed. Keep it local for a
  # one-person project, or move it to an OCI Object Storage bucket (S3-
  # compatible backend) once more than one person runs apply.
}

provider "oci" {
  tenancy_ocid     = var.tenancy_ocid
  user_ocid        = var.user_ocid
  fingerprint      = var.fingerprint
  private_key_path = var.private_key_path
  region           = var.region
}
