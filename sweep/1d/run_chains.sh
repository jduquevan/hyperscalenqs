for HAM in ising heisenberg j1j2; do
    # sbatch sweep/1d/run_minsr.slurm $HAM minsr
    sbatch sweep/1d/run_minsr.slurm $HAM spring
    sbatch sweep/1d/run_adam.slurm $HAM
    sbatch sweep/1d/run_qps.slurm $HAM
done