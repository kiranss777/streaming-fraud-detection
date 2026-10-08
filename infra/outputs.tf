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
