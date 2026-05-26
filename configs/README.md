# XANESNET Example Configurations

This directory holds reference YAML configurations for the three XANESNET
workflows: training, checkpointed inference, and prediction analysis. Each file
is validated by the packaged JSON Schemas under
[../xanesnet/schemas/](../xanesnet/schemas/) and may be used directly with the
`xanesnet` command-line tool.

## How configs are used

A config drives one XANESNET run. The supported workflow today is forward
prediction from structure to spectra:

1. **Train** — `xanesnet train -i <file>` builds the datasource, dataset,
   model, trainer, and training strategy described by the config. Outputs are
   written under `runs/<timestamp>_train_<name>/`.
2. **Infer** — `xanesnet infer -i <file> -m <checkpoint>` merges the user
   config with the checkpoint signature before validation. Predictions land in
   `runs/<timestamp>_infer_<name>/predictions/`.
3. **Analyze** — `xanesnet analyze -i <file> -p <predictions>` computes and
   reports metrics on saved predictions.

All schema defaults are materialized in place during validation, so example
configs only spell out fields that differ from the schema defaults.

## Authoring tips

- Use the interactive editor in [../tools/config-ui/](../tools/config-ui/) to
  browse valid fields and generate new configs.
- A training config may set selected top-level `model` fields to `auto`; they
  are resolved from a prepared dataset sample at run time.
- For inference, leave fields that the checkpoint signature owns
  (`dataset_type`, `model`, `strategy`) out of the user config unless an
  override is intentional. Conflicting user values cause early failure.
- Validate a new config locally without running training:

  ```bash
  python -c "from xanesnet.serialization.config import load_raw_config, validate_config_train; \
             validate_config_train(load_raw_config('configs/in_mlp.yaml'))"
  ```
