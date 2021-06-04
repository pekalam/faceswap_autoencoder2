#!/bin/bash

docker-compose -f ./e2e/training_logs/docker-compose.yml build
docker-compose -f ./e2e/prod_config/docker-compose.yml build
