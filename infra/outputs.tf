output "bucket_name" {
  value = aws_s3_bucket.data.bucket
}

output "glue_role_arn" {
  value = aws_iam_role.this["glue"].arn
}

output "glue_etl_job" {
  value = aws_glue_job.etl_features.name
}

output "sagemaker_role_arn" {
  value = aws_iam_role.this["sagemaker"].arn
}

output "training_pipeline" {
  value = aws_sagemaker_pipeline.training.pipeline_name
}

output "instance_ids" {
  value = { for role, h in aws_instance.host : role => h.id }
}

output "connect" {
  description = "Open a shell on a host (needs the Session Manager plugin)"
  value       = { for role, h in aws_instance.host : role => "aws ssm start-session --target ${h.id}" }
}

output "dashboard_url" {
  description = "Reachable only from dashboard_cidr, when set"
  value       = "http://${aws_instance.host["compute"].public_ip}:8501"
}
