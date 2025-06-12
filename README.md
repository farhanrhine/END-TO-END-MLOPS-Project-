# Jenkins with Docker and AWS Deployment

## Overview
This README provides a comprehensive guide to setting up a Jenkins instance with Docker-in-Docker (DinD), installing necessary tools, building and deploying Docker images, and integrating AWS CLI for deployment.

---

## Dockerfile

```Dockerfile
# Use the Jenkins image as the base image
FROM jenkins/jenkins:lts

# Switch to root user to install dependencies
USER root

# Install prerequisites and Docker
RUN apt-get update -y && \
    apt-get install -y apt-transport-https ca-certificates curl gnupg software-properties-common && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add - && \
    echo "deb [arch=amd64] https://download.docker.com/linux/debian bullseye stable" > /etc/apt/sources.list.d/docker.list && \
    apt-get update -y && \
    apt-get install -y docker-ce docker-ce-cli containerd.io && \
    apt-get clean

# Add Jenkins user to the Docker group (create if it doesn't exist)
RUN groupadd -f docker && \
    usermod -aG docker jenkins

# Create the Docker directory and volume for DinD
RUN mkdir -p /var/lib/docker
VOLUME /var/lib/docker

# Switch back to the Jenkins user
USER jenkins
```

---

## Commands

### Build Docker Image
```bash
docker build -t jenkins-dind .
```

### Kill Process on Port 8080 (Windows)
```bash
netstat -ano | find ":8080"            taskkill /PID <PID> /F
```

### Run Jenkins with Docker-in-Docker
```bash
docker run -d --name jenkins-dind ^
--privileged ^
-p 8080:8080 -p 50000:50000 ^
-v //var/run/docker.sock:/var/run/docker.sock ^
-v jenkins_home:/var/jenkins_home ^
jenkins-dind
```
```
or for powershell
docker run -d --name jenkins-dind --privileged -p 8080:8080 -p 50000:50000 -v //var/run/docker.sock:/var/run/docker.sock -v jenkins_home:/var/jenkins_home jenkins-dind
```

### Install Python and Tools in the Container
```bash
docker exec -u root -it <container_name_or_id> apt update -y
docker exec -u root -it jenkins-dind apt install -y python3
docker exec -u root -it jenkins-dind python3 --version
docker exec -u root -it jenkins-dind ln -s /usr/bin/python3 /usr/bin/python
docker exec -u root -it jenkins-dind python --version
docker exec -u root -it jenkins-dind apt install -y python3-pip
docker exec -u root -it jenkins-dind apt install -y python3-venv
```

---

## Jenkinsfile

```groovy
pipeline {
    agent any
    
    stages {
        stage('Build Docker image') {
            steps {
                script {
                    echo 'Build Docker image...'
                    docker.build("mlops")
                }
            }
        }
    }
}
```

---

## Trivy FileSystem Scan

### Install Trivy
```bash
docker exec -u root -it jenkins-dind bash
curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin v0.57.1
```

---

## Additional Steps

### Add Jenkins User to Docker Group
```bash
docker exec -u root -it jenkins-dind bash
groupadd docker
usermod -aG docker jenkins
usermod -aG root Jenkins
```

---

## Sample Dockerfile for Python Application

```Dockerfile
# Use a lightweight Python image
FROM python:slim

# Set environment variables to prevent Python from writing .pyc files & Ensure Python output is not buffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the application code
COPY . .

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Train the model before running the application
RUN python main.py

# Expose the port that Flask will run on
EXPOSE 5000

# Command to run the app
CMD ["python", "application.py"]
```

---

## AWS Deployment

### Configure AWS CLI
```bash
aws configure
```

### Install AWS CLI in the Container
```bash
docker exec -u root -it jenkins-dind bash
aws --version
apt update && apt upgrade -y
apt-get install curl unzip sudo -y
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
aws --version
exit
docker restart jenkins-dind
```

### Update ECS Service
```bash
sh "aws ecs update-service --cluster clustername --service servicename --force-new-deployment"
```

---

## Summary
This setup provides:
- A Jenkins environment integrated with Docker.
- Trivy for filesystem scanning.
- AWS CLI for deployment.

Follow the above steps carefully to ensure a smooth setup and deployment experience.

