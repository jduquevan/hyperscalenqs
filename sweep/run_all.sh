for HAM in ising heisenberg j1j2; do
    # sbatch sweep/run_minsr_a100.slurm $HAM minsr
    sbatch sweep/run_minsr_a100.slurm $HAM spring
    sbatch sweep/run_adam_a100.slurm $HAM
    sbatch sweep/run_qps_a100.slurm $HAM
done