# Read/write on the data bucket, shared by Glue and SageMaker.
resource "aws_iam_policy" "bucket_access" {
  name = "${var.project}-bucket-access"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = aws_s3_bucket.data.arn
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
        Resource = "${aws_s3_bucket.data.arn}/*"
      },
    ]
  })
}

locals {
  # role name => AWS service that assumes it, and the AWS-managed policy it gets
  roles = {
    glue = {
      service = "glue.amazonaws.com"
      policy  = "arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
    }
    sagemaker = {
      service = "sagemaker.amazonaws.com"
      policy  = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
    }
  }
}

resource "aws_iam_role" "this" {
  for_each = local.roles
  name     = "${var.project}-${each.key}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = each.value.service }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "managed" {
  for_each   = local.roles
  role       = aws_iam_role.this[each.key].name
  policy_arn = each.value.policy
}

resource "aws_iam_role_policy_attachment" "bucket" {
  for_each   = local.roles
  role       = aws_iam_role.this[each.key].name
  policy_arn = aws_iam_policy.bucket_access.arn
}
