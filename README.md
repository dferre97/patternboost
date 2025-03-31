# About this repo

Working and improved re-implementation of the PatternBoost method (https://github.com/zawagner22/transformers_math_experiments), work in progress.

# How to run the container on Windows

1. Start Docker Desktop
2. Access the WLS
3. docker run --runtime=nvidia --volume=/home/farcitoast/projects/my_patternboost:/root/projects/my_patternboost --workdir=/root/projects --name my_patternboost -t -d --gpus all python-julia-cuda:latest
4. The container my_patternboost should now run in the background (check in Docker Desktop), and you can attach to it using vscode 
