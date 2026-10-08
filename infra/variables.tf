variable "region" {
  type    = string
  default = "us-east-1"
}

variable "project" {
  type    = string
  default = "streaming-fraud-detection"
}

variable "instance_type" {
  description = "Size of each of the two hosts (data, compute). Largest type the account's Free plan allows."
  type        = string
  default     = "m7i-flex.large" # 2 vCPU, 8 GB each
}
