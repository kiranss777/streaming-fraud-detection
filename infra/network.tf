# One public subnet. Hosts get public IPs for OUTBOUND traffic only (pulling images,
# reaching S3/SSM). Nothing on the internet can connect in: the only inbound rules let the
# two hosts talk to each other, and humans get in through SSM Session Manager.

data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_vpc" "main" {
  cidr_block           = "10.20.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = { Name = var.project }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.20.1.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

resource "aws_security_group" "host" {
  name        = "${var.project}-host"
  description = "Pipeline hosts - inbound only from each other, access via SSM"
  vpc_id      = aws_vpc.main.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# The only inbound traffic allowed: the two hosts talking to each other (self = same group).
resource "aws_vpc_security_group_ingress_rule" "kafka_between_hosts" {
  security_group_id            = aws_security_group.host.id
  referenced_security_group_id = aws_security_group.host.id
  ip_protocol                  = "tcp"
  from_port                    = 9092
  to_port                      = 9092
  description                  = "Kafka, compute host to data host"
}

resource "aws_vpc_security_group_ingress_rule" "cassandra_between_hosts" {
  security_group_id            = aws_security_group.host.id
  referenced_security_group_id = aws_security_group.host.id
  ip_protocol                  = "tcp"
  from_port                    = 9042
  to_port                      = 9042
  description                  = "Cassandra, compute host to data host"
}
