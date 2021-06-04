#!/bin/bash
ARGS=( "$@" )

for (( i=0; i<${#ARGS[@]}; i+=2 )); do

declare "${ARGS[i]}"="${ARGS[i+1]}"
export ${ARGS[i]}

done


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/root/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/root/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/root/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate ml1

if ! [ -e "/root/.local/share/jupyter/kernels/ml1" ]; then
    echo "installing ml1 kernel"
    python -m ipykernel install --user --name ml1 --display-name "Python (conda ml1)"
fi


exec jupyter notebook --allow-root --ip=0.0.0.0 --notebook-dir=/app