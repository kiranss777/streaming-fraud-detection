# Training is a one-off run, so Terraform doesn't run it - it defines the recipe
# (a SageMaker Pipeline) and the trigger (EventBridge: Glue ETL succeeded -> train).

# ---------- Training code ----------
# Uploaded as a plain S3 folder and mounted into the container as the "code" input.
# The container's toolkit accepts a local folder as the code location, so no tar.gz needed.
locals {
  training_files = ["train.py", "requirements.txt"]
}

resource "aws_s3_object" "training_code" {
  for_each = toset(local.training_files)
  bucket   = aws_s3_bucket.data.id
  key      = "scripts/training/${each.value}"
  source   = "${path.module}/../training/${each.value}"
  etag     = filemd5("${path.module}/../training/${each.value}")
}

# AWS-managed XGBoost container (XGBoost 3.0, Python 3.10).
data "aws_sagemaker_prebuilt_ecr_image" "xgboost" {
  repository_name = "sagemaker-xgboost"
  image_tag       = "3.0-5"
}

# ---------- The recipe ----------
resource "aws_sagemaker_pipeline" "training" {
  pipeline_name         = "${var.project}-training"
  pipeline_display_name = "${var.project}-training"
  role_arn              = aws_iam_role.this["sagemaker"].arn

  pipeline_definition = jsonencode({
    Version = "2020-12-01"
    Steps = [{
      Name = "TuneFraudModel"
      Type = "Tuning"
      # Same fields as the CreateHyperParameterTuningJob API.
      Arguments = {
        HyperParameterTuningJobConfig = {
          # Bayesian: learns from finished runs which settings look promising and tries those next.
          Strategy = "Bayesian"
          HyperParameterTuningJobObjective = {
            Type       = "Maximize"
            MetricName = "validation:aucpr" # built-in XGBoost metric, validation set only - test stays untouched
          }
          ResourceLimits = {
            MaxNumberOfTrainingJobs = 20
            MaxParallelTrainingJobs = 4
          }
          ParameterRanges = {
            IntegerParameterRanges = [
              { Name = "max_depth", MinValue = "3", MaxValue = "10", ScalingType = "Auto" },
            ]
            ContinuousParameterRanges = [
              { Name = "eta", MinValue = "0.02", MaxValue = "0.3", ScalingType = "Logarithmic" },
              { Name = "min_child_weight", MinValue = "1", MaxValue = "20", ScalingType = "Logarithmic" },
              { Name = "subsample", MinValue = "0.6", MaxValue = "1.0", ScalingType = "Linear" },
              { Name = "colsample_bytree", MinValue = "0.5", MaxValue = "1.0", ScalingType = "Linear" },
              { Name = "max_delta_step", MinValue = "0", MaxValue = "10", ScalingType = "Linear" },
            ]
          }
          TrainingJobEarlyStoppingType = "Off"
        }

        # What each of the 20 trial runs looks like (CreateTrainingJob fields).
        TrainingJobDefinition = {
          RoleArn = aws_iam_role.this["sagemaker"].arn
          AlgorithmSpecification = {
            TrainingImage     = data.aws_sagemaker_prebuilt_ecr_image.xgboost.registry_path
            TrainingInputMode = "File"
          }
          # Fixed for every trial. Script-mode values are JSON-encoded, as the container expects.
          StaticHyperParameters = {
            sagemaker_program          = jsonencode("train.py")
            sagemaker_submit_directory = jsonencode("/opt/ml/input/data/code")
            num_round                  = "1000"
            early_stopping_rounds      = "50"
          }
          InputDataConfig = [
            for name, prefix in {
              train = "processed/train/"
              test  = "processed/test/"
              code  = "scripts/training/"
              } : {
              ChannelName = name
              DataSource = {
                S3DataSource = {
                  S3DataType             = "S3Prefix"
                  S3Uri                  = "s3://${aws_s3_bucket.data.bucket}/${prefix}"
                  S3DataDistributionType = "FullyReplicated"
                }
              }
            }
          ]
          OutputDataConfig = { S3OutputPath = "s3://${aws_s3_bucket.data.bucket}/models/" }
          ResourceConfig = {
            InstanceCount  = 1
            InstanceType   = "ml.m5.large"
            VolumeSizeInGB = 10
          }
          StoppingCondition = { MaxRuntimeInSeconds = 3600 } # hard stop at 1h per trial
        }
      }
    }]
  })
}

# The pipeline (running as the SageMaker role) hands that same role to the training job.
resource "aws_iam_role_policy" "sagemaker_pass_self" {
  role = aws_iam_role.this["sagemaker"].name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "iam:PassRole"
      Resource  = aws_iam_role.this["sagemaker"].arn
      Condition = { StringEquals = { "iam:PassedToService" = "sagemaker.amazonaws.com" } }
    }]
  })
}

# ---------- The trigger: Glue ETL succeeded -> start training ----------
resource "aws_cloudwatch_event_rule" "etl_succeeded" {
  name        = "${var.project}-etl-succeeded"
  description = "Start model training when the Glue ETL job succeeds"
  event_pattern = jsonencode({
    source      = ["aws.glue"]
    detail-type = ["Glue Job State Change"]
    detail = {
      jobName = [aws_glue_job.etl_features.name]
      state   = ["SUCCEEDED"]
    }
  })
}

resource "aws_cloudwatch_event_target" "start_training" {
  rule     = aws_cloudwatch_event_rule.etl_succeeded.name
  arn      = aws_sagemaker_pipeline.training.arn
  role_arn = aws_iam_role.eventbridge.arn
}

# EventBridge's badge: allowed to start this one pipeline, nothing else.
resource "aws_iam_role" "eventbridge" {
  name = "${var.project}-eventbridge"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "events.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "eventbridge_start_pipeline" {
  role = aws_iam_role.eventbridge.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "sagemaker:StartPipelineExecution"
      Resource = aws_sagemaker_pipeline.training.arn
    }]
  })
}
