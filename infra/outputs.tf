output "bucket_name" {
  value = aws_s3_bucket.data.bucket
}

output "glue_role_arn" {
  value = aws_iam_role.this["glue"].arn
}

output "sagemaker_role_arn" {
  value = aws_iam_role.this["sagemaker"].arn
}
