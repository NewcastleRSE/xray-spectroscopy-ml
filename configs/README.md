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
   encodings, model, trainer, and training strategy described by the config.
   Outputs are written under `runs/<timestamp>_train_<name>/`.
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
- The `encodings` list defines an ordered pipeline that maps raw target spectra
  into the space the model predicts in. Encodings are applied in order during
  training and inverted in reverse order to decode predictions back to spectra.
  Use `- encoding_type: identity` for raw-spectrum training, or compose entries
  such as `scale`, `fourier`, `gaussian`, `z_score`, `min_max`, and
  `subtract_average`. The `z_score`, `min_max`, and `scale` encodings accept a
  `per_point` flag (default `true`) that switches between per-point statistics
  and a single global statistic. The `z_score`, `min_max`, `scale`, and
  `subtract_average` encodings also accept a `per_element` flag (default
  `false`); when `true` they select their statistics per absorbing element via a
  parallel `elements` list of atomic numbers, with one parameter row per
  element. Dataset-dependent parameters may be
  set to `auto` and are resolved from the training data: Gaussian `num_points`,
  the `z_score` `mean`/`std`, the `min_max` `minimum`/`maximum`, the `scale`
  `factor`, the `subtract_average` `average`, and (when `per_element` is `true`)
  the `elements` list plus their per-element parameters. The encoding pipeline is
  stored in the checkpoint signature and restored at inference.
- For inference, leave fields that the checkpoint signature owns
  (`dataset_type`, `model`, `strategy`, `encodings`) out of the user config
  unless an override is intentional. Conflicting user values cause early
  failure.
- Validate a new config locally without running training:

  ```bash
  python -c "from xanesnet.serialization.config import load_raw_config, validate_config_train; \
             validate_config_train(load_raw_config('configs/in_mlp.yaml'))"
  ```
