#!/bin/bash
set -e

pushd /

branch="$1"
token="$2"

if [ -v "$branch" ] || [ -v "$token" ]; then
    echo "Branch and token args not provided"
    exit 1
fi

echo "Cloning branch $1"

if [ -e "faceswap_autoencoder" ]; then
    echo "Removing exisitng repo"
    rm -rf ./faceswap_autoencoder
fi

git clone -b "$branch" "https://$token@github.com/pekalam/faceswap_autoencoder.git"

to_cpy=( "src" "experiments" "__dataset" "__dataset2_13" "__dataset2_13-rev1" "__dataset3" "__dataset3_masked" "__dataset3_masked_large" )

for f in "${to_cpy[@]}"; do
    path="/app/$f"
    repo_path="/faceswap_autoencoder/$f"
    if [ -e "$path" ]; then
        echo "Removing existing $path" 
        rm -rf "$path"
    fi
    cp -r "$repo_path" "/app"
done




popd || exit