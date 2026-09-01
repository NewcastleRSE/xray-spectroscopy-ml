# Paper workflow

The paper workflow configurations live in
[configs/paper_workflow/](../../configs/paper_workflow/).

- [train_paper_workflow.sh](train_paper_workflow.sh)
- [infer_paper_workflow.sh](infer_paper_workflow.sh)
- [analyze_paper_workflow.sh](analyze_paper_workflow.sh)
- [run_usage_workflows.sh](run_usage_workflows.sh)

Run a stage independently after its prerequisites are available:

```bash
bash scripts/paper_workflow/train_paper_workflow.sh
bash scripts/paper_workflow/infer_paper_workflow.sh
bash scripts/paper_workflow/analyze_paper_workflow.sh
```

All dispatchers use `runs/paper_workflow/` by default and use the consistent
model names `paper_workflow_mlp` and `paper_workflow_schnet`. Set `OUT_DIR` to
use another output location. Inference also accepts `MLP_MODEL` and
`SCHNET_MODEL`; analysis accepts `MLP_RUN` and `SCHNET_RUN` when explicit paths
are needed.