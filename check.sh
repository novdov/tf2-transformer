#!/usr/bin/env bash

echo "[ run black ]"
black -v transformer/

echo "[ run flake8 ]"
flake8 transformer/

echo "[ run isort ]"
isort -rc transformer/
