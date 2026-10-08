# Two hosts (the account's Free plan caps instance size, so the stack is split by tier):
#   data    - Kafka + Cassandra
#   compute - Spark streaming job, producer, Streamlit dashboard
# Same total as one 4 vCPU / 16 GB box, at the same price, and storage is separated from compute.

# ---------- App files the hosts run ----------
# Same pattern as the Glue and training code: git -> Terraform -> S3 -> the hosts pull it.
locals {
  app_files = concat(
    ["docker-compose.yml", "cassandra/schema.cql"],
    tolist(fileset("${path.module}/..", "producer/*")),
    tolist(fileset("${path.module}/..", "stream/*.{py,txt}")),
    ["stream/Dockerfile"],
    ["dashboard/app.py", "dashboard/requirements.txt", "dashboard/Dockerfile"],
  )

  hosts = {
    data    = { private_ip = "10.20.1.10", disk_gb = 30 } # Kafka log + Cassandra data
    compute = { private_ip = "10.20.1.11", disk_gb = 20 }
  }
}

resource "aws_s3_object" "app" {
  for_each = toset(local.app_files)
  bucket   = aws_s3_bucket.data.id
  key      = "app/${each.value}"
  source   = "${path.module}/../${each.value}"
  etag     = filemd5("${path.module}/../${each.value}")
}

# ---------- The hosts ----------
data "aws_ssm_parameter" "al2023" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

resource "aws_instance" "host" {
  for_each = local.hosts

  ami                    = data.aws_ssm_parameter.al2023.value
  instance_type          = var.instance_type
  subnet_id              = aws_subnet.public.id
  private_ip             = each.value.private_ip # fixed, so each host knows where the other is
  vpc_security_group_ids = [aws_security_group.host.id]
  iam_instance_profile   = aws_iam_instance_profile.host.name

  root_block_device {
    volume_size = each.value.disk_gb
    volume_type = "gp3"
    encrypted   = true
  }

  metadata_options {
    http_tokens                 = "required" # IMDSv2 only
    http_put_response_hop_limit = 2          # +1 hop so Docker containers can use the host's IAM role
  }

  # First boot: install Docker + Compose, pull the app files from S3, start this host's
  # services (compose profile = host role). Later updates go out via `aws ssm send-command`.
  user_data = <<-EOF
    #!/bin/bash
    set -euxo pipefail
    dnf install -y docker
    systemctl enable --now docker
    mkdir -p /usr/local/lib/docker/cli-plugins
    curl -fsSL https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-x86_64 \
      -o /usr/local/lib/docker/cli-plugins/docker-compose
    chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
    mkdir -p /opt/fraud
    aws s3 sync s3://${aws_s3_bucket.data.bucket}/app/ /opt/fraud/
    cat > /opt/fraud/.env <<'ENV'
    COMPOSE_PROFILES=${each.key}
    DATA_HOST_IP=${local.hosts.data.private_ip}
    DATA_BUCKET=${aws_s3_bucket.data.bucket}
    AWS_DEFAULT_REGION=${var.region}
    ENV
    cd /opt/fraud && docker compose up -d
  EOF

  # A newer AMI shouldn't replace a running host and wipe Kafka/Cassandra data.
  lifecycle {
    ignore_changes = [ami]
  }

  depends_on = [aws_s3_object.app] # files must be in S3 before first boot pulls them

  tags = { Name = "${var.project}-${each.key}" }
}
