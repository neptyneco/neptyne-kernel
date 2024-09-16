#!/bin/bash

echo "$0" env: "$(env)"

# Large ulimit nofile values makes subprocesses slow.
ulimit -n 1024

launch_python_kernel() {
    export JPY_PARENT_PID=$$  # Force reset of parent pid since we're detached

	  set -x
	  python -m neptyne_kernel.launch_ipykernel \
	      --key "$KERNEL_KEY"
	  { set +x; } 2>/dev/null
}

if [ -z "${KERNEL_KEY+x}" ]
then
    echo "Environment variable KERNEL_KEY is required."
    exit 1
fi

launch_python_kernel
