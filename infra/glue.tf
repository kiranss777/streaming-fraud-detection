# The ETL script lives in git (glue/etl_features.py); Terraform uploads it to S3
# and re-uploads whenever the file changes (etag).
resource "aws_s3_object" "etl_script" {
  bucket = aws_s3_bucket.data.id
  key    = "scripts/etl_features.py"
  source = "${path.module}/../glue/etl_features.py"
  etag   = filemd5("${path.module}/../glue/etl_features.py")
}

resource "aws_glue_job" "etl_features" {
  name     = "${var.project}-etl-features"
  role_arn = aws_iam_role.this["glue"].arn

  glue_version      = "5.0"
  worker_type       = "G.1X" # 4 vCPU / 16 GB per worker
  number_of_workers = 2      # the minimum
  timeout           = 30     # minutes - kill runaway jobs before they cost much

  command {
    name            = "glueetl"
    script_location = "s3://${aws_s3_bucket.data.bucket}/${aws_s3_object.etl_script.key}"
    python_version  = "3"
  }

  default_arguments = {
    "--DATA_BUCKET"                      = aws_s3_bucket.data.bucket
    "--enable-continuous-cloudwatch-log" = "true"
  }
}
