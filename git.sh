#!/bin/sh

TODAY=$(date +"%y-%m-%d")

git add .
git commit -m "$TODAY"
git push