export N_SAMPLES=1024 CHAIN_N=12 N_GRU_LAYERS=3 RNN_HIDDEN=256 HEAD_HIDDEN=256
export WANDB_PROJECT=chains-corrected

for HAM in ising heisenberg j1j2; do
    if [ "$HAM" = "ising" ]; then
        ADAM_STEPS=5000
        PWO_STEPS=20000
    else
        ADAM_STEPS=10000
        PWO_STEPS=40000
    fi

    sbatch sweep/run_minsr.slurm "$HAM" minsr
    sbatch sweep/run_minsr.slurm "$HAM" spring
    TRANSITION_STEPS=$ADAM_STEPS sbatch sweep/run_adam.slurm "$HAM"
    PPO_EPOCHS=4 TRANSITION_STEPS=$PWO_STEPS sbatch sweep/run_pwo.slurm "$HAM"
done