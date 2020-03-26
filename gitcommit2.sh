#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -m commit_message"
   echo -e "\t-m added a message to the commit (mandatory)"
   exit 1 # Exit script after printing help
}

while getopts "m:" opt
do
   case "$opt" in
      m ) parameterM="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$parameterM" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

coverage run -m pytest
coverage html -d Coverage
rm Coverage/coverage.svg
coverage-badge -o Coverage/coverage.svg
git add .github/
git add Coverage/
git commit -m "$parameterM"