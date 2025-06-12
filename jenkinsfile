pipeline {
    agent any
    
    stages {
        stage('Cloning from Github Repo') {
            steps {
                script {
                    // Cloning Github repo
                    echo 'Cloning from Github Repo.........'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'mlops github token', url: 'https://github.com/farhanrhine/END-TO-END-MLOPS-Project-.git']])
                }
            }
        }

    }
}
