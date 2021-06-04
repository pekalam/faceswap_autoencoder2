#!/bin/bash


#!!!!!!!!!!
#WARNING - overrides existing
#!!!!!!!!!!
#echo "Copying base e2e docker files..."
#sleep 3
#cp -rf ../Dockerfile.e2e ../docker-compose.yml ./

rm -rf ./outputs

EXPECTED_RUNS=2


docker-compose up $1

actual_runs=$(ls ./outputs/*/*/ray_results/*/ | grep TuneTrainer | wc -l)

if ! [ $actual_runs -eq $EXPECTED_RUNS ]; then
    echo "Found $actual_runs runs expected: $EXPECTED_RUNS"
    exit 1
fi


#test artifact files
mlruns=( $(ls ./outputs/*/*/mlruns/*/*/artifacts | sed '/\:/d') )
ray_results=( $(ls ./outputs/*/*/ray_results/*/Tune* | sed '/\:/d') )

for f1 in "${mlruns[@]}"; do
    match=0
    if [ "$f1" == "test.png" ]; then
        continue
    fi
    for f2 in "${ray_results[@]}"; do
        if [ "$f1" == "$f2" ]; then
            match=1
            break
        fi
    done
    if [ "$match" -eq 0 ]; then
        echo "Files in mlruns not matching with ray_results"
        echo "mlruns:"
        ls ./outputs/*/*/mlruns/*/*/artifacts
        echo "ray_results"
        ls ./outputs/*/*/ray_results/*/Tune*
        exit 1
    fi
done


#test checkpoint files
mlruns_ckpt=("$(ls ./outputs/*/*/mlruns/*/*/artifacts/checkpoint* | sed '/\:/d')")
ray_results_ckpt=("$(ls ./outputs/*/*/ray_results/*/Tune*/checkpoint* | sed '/\:/d')")

for f1 in "${mlruns_ckpt[@]}"; do
    match=0
    for f2 in "${ray_results_ckpt[@]}"; do
        if [ "$f1" == "$f2" ]; then
            match=1
            break
        fi
    done
    if [ "$match" -eq 0 ]; then
        echo "Files in mlruns checkpoint not matching with ray_results checkpoint"
        echo "mlruns:"
        ls ./outputs/*/*/mlruns/*/*/artifacts/checkpoint*
        echo "ray_results"
        ls ./outputs/*/*/ray_results/*/Tune*/checkpoint*
        exit 1
    fi
done


#python ./src/checkpoint_tester.py D:\\WIN10_SSDSamsung\\TEMP\\jupyter-outputs\\gradient_arch_batch_dataset2_best_perf\\jupyter-outputs\\ray_results\\arch_batch_dataset2\\TuneTrainer_6cba2_00000 checkpoint_000226

if [ -n "$NO_CHECKPOINT_TEST" ]; then
    echo "skipping checkpoint test"

else
    echo "starting checkpoint test"
    CHECKPOINTS=( $(find . -path \*/ray_results\* -name "checkpoint*") )

    for checkpoint in "${CHECKPOINTS[@]}"; do
        echo "testing checkpoint $checkpoint"
        dir_split=( ${checkpoint//// } )
        python ../../src/checkpoint_tester.py "$(dirname "$checkpoint")" "${dir_split[-1]}"
    done
fi

if [ -n "$NO_TEARDOWN" ]; then
    echo "skipping teardown"
else
    echo "teardown"
    bash ./teardown.sh
fi