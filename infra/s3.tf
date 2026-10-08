data "aws_caller_identity" "current" {}

# raw/ (Kaggle CSVs), processed/ (Glue output), models/ (SageMaker artifacts)
resource "aws_s3_bucket" "data" {
  bucket = "fraud-pipeline-${data.aws_caller_identity.current.account_id}-${var.region}"

  # Everything in here is reproducible from Kaggle, so let `terraform destroy` empty it.
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket = aws_s3_bucket.data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
