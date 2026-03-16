# Experiment Record: YYYYMMDD_HHMMSS

## Run Summary
- Pipeline: `pinn` or `deepsets`
- Timestamp: `YYYYMMDD_HHMMSS`
- Config file: `configs/xxx.json`
- Transform skipped: `false`
- Train skipped: `false`

## Inputs
- Noisy mat: `data/noisy.mat`
- Clean mat: `data/clean.mat`
- Inference input: `data/noisy.mat`

## Outputs
- Checkpoint: `results/checkpoints/best_xxx.pth`
- Denoised mat: `results/xxx.mat`
- Acoustic validation figure: `results/images/xxx.png`

## Full Runtime Config
```json
{
  "pipeline": "pinn",
  "epochs": 50,
  "batch_size": 32,
  "lr": 0.001,
  "physics_weight": 0.001
}
```

## Metrics (Fill After Review)
- Best val PSNR:
- Final train PSNR:
- Physics loss trend:
- Acoustic validation verdict:

## Notes
- What changed vs previous run:
- Observed failure modes:
- Next run plan:
