#!/bin/bash
# This script was converted from a main.yml GitHub workflow file.
set -e

# --- MLP Models ---

echo "--- Running: MLP (STD) ---"
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/mlp_std_xyz_001 --in_file ./.github/workflows/inputs/in_predict.yaml
python3 -m xanesnet.cli --mode train_xanes --in_file ./.github/workflows/inputs/in_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xyz --in_model models/mlp_std_xanes_001 --in_file ./.github/workflows/inputs/in_predict.yaml

echo "--- Running: MLP (Kfold) ---"
sed -i 's/kfold: False/kfold: True/' ./.github/workflows/inputs/in_mlp.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/mlp_kfold_xyz_001 --in_file ./.github/workflows/inputs/in_predict.yaml
python3 -m xanesnet.cli --mode train_xanes --in_file ./.github/workflows/inputs/in_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xyz --in_model models/mlp_kfold_xanes_001 --in_file ./.github/workflows/inputs/in_predict.yaml
sed -i 's/kfold: True/kfold: False/' ./.github/workflows/inputs/in_mlp.yaml

echo "--- Running: MLP (Bootstrap) ---"
sed -i 's/bootstrap: False/bootstrap: True/' ./.github/workflows/inputs/in_mlp.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/mlp_bootstrap_xyz_001 --in_file ./.github/workflows/inputs/in_predict.yaml
python3 -m xanesnet.cli --mode train_xanes --in_file ./.github/workflows/inputs/in_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xyz --in_model models/mlp_bootstrap_xanes_001 --in_file ./.github/workflows/inputs/in_predict.yaml
sed -i 's/bootstrap: True/bootstrap: False/' ./.github/workflows/inputs/in_mlp.yaml

echo "--- Running: MLP (Ensemble) ---"
sed -i 's/ensemble: False/ensemble: True/' ./.github/workflows/inputs/in_mlp.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/mlp_ensemble_xyz_001 --in_file ./.github/workflows/inputs/in_predict.yaml
python3 -m xanesnet.cli --mode train_xanes --in_file ./.github/workflows/inputs/in_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xyz --in_model models/mlp_ensemble_xanes_001 --in_file ./.github/workflows/inputs/in_predict.yaml
sed -i 's/ensemble: True/ensemble: False/' ./.github/workflows/inputs/in_mlp.yaml

# --- Other Architectures (STD) ---

echo "--- Running: CNN (STD) ---"
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_cnn.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/cnn_std_xyz_001 --in_file ./.github/workflows/inputs/in_predict.yaml
python3 -m xanesnet.cli --mode train_xanes --in_file ./.github/workflows/inputs/in_cnn.yaml --save
python3 -m xanesnet.cli --mode predict_xyz --in_model models/cnn_std_xanes_001 --in_file ./.github/workflows/inputs/in_predict.yaml

echo "--- Running: LSTM (STD) ---"
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_lstm.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/lstm_std_xyz_001 --in_file ./.github/workflows/inputs/in_predict.yaml
python3 -m xanesnet.cli --mode train_xanes --in_file ./.github/workflows/inputs/in_lstm.yaml --save
python3 -m xanesnet.cli --mode predict_xyz --in_model models/lstm_std_xanes_001 --in_file ./.github/workflows/inputs/in_predict.yaml

# --- Autoencoder Models ---

echo "--- Running: AE_MLP (STD) ---"
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_ae_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/ae_mlp_std_xyz_001 --in_file ./.github/workflows/inputs/in_predict.yaml
python3 -m xanesnet.cli --mode train_xanes --in_file ./.github/workflows/inputs/in_ae_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xyz --in_model models/ae_mlp_std_xanes_001 --in_file ./.github/workflows/inputs/in_predict.yaml

echo "--- Running: AE_MLP (Kfold) ---"
sed -i 's/kfold: False/kfold: True/' ./.github/workflows/inputs/in_ae_mlp.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_ae_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/ae_mlp_kfold_xyz_001 --in_file ./.github/workflows/inputs/in_predict.yaml
python3 -m xanesnet.cli --mode train_xanes --in_file ./.github/workflows/inputs/in_ae_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xyz --in_model models/ae_mlp_kfold_xanes_001 --in_file ./.github/workflows/inputs/in_predict.yaml
sed -i 's/kfold: True/kfold: False/' ./.github/workflows/inputs/in_ae_mlp.yaml

echo "--- Running: AE_MLP (Bootstrap) ---"
sed -i 's/bootstrap: False/bootstrap: True/' ./.github/workflows/inputs/in_ae_mlp.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_ae_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/ae_mlp_bootstrap_xyz_001 --in_file ./.github/workflows/inputs/in_predict.yaml
python3 -m xanesnet.cli --mode train_xanes --in_file ./.github/workflows/inputs/in_ae_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xyz --in_model models/ae_mlp_bootstrap_xanes_001 --in_file ./.github/workflows/inputs/in_predict.yaml
sed -i 's/bootstrap: True/bootstrap: False/' ./.github/workflows/inputs/in_ae_mlp.yaml

echo "--- Running: AE_MLP (Ensemble) ---"
sed -i 's/ensemble: False/ensemble: True/' ./.github/workflows/inputs/in_ae_mlp.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_ae_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/ae_mlp_ensemble_xyz_001 --in_file ./.github/workflows/inputs/in_predict.yaml
python3 -m xanesnet.cli --mode train_xanes --in_file ./.github/workflows/inputs/in_ae_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xyz --in_model models/ae_mlp_ensemble_xanes_001 --in_file ./.github/workflows/inputs/in_predict.yaml
sed -i 's/ensemble: True/ensemble: False/' ./.github/workflows/inputs/in_ae_mlp.yaml

echo "--- Running: AE_CNN (STD) ---"
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_ae_cnn.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/ae_mlp_std_xyz_001 --in_file ./.github/workflows/inputs/in_predict.yaml
python3 -m xanesnet.cli --mode train_xanes --in_file ./.github/workflows/inputs/in_ae_cnn.yaml --save
python3 -m xanesnet.cli --mode predict_xyz --in_model models/ae_cnn_std_xanes_001 --in_file ./.github/workflows/inputs/in_predict.yaml

# --- AEGAN Models ---

echo "--- Running: AEGAN (STD) ---"
python3 -m xanesnet.cli --mode train_all --in_file ./.github/workflows/inputs/in_aegan.yaml --save
python3 -m xanesnet.cli --mode predict_all --in_model models/aegan_mlp_std_all_001 --in_file ./.github/workflows/inputs/in_predict.yaml

echo "--- Running: AEGAN (Kfold) ---"
sed -i 's/kfold: False/kfold: True/' ./.github/workflows/inputs/in_aegan.yaml
python3 -m xanesnet.cli --mode train_all --in_file ./.github/workflows/inputs/in_aegan.yaml --save
python3 -m xanesnet.cli --mode predict_all --in_model models/aegan_mlp_kfold_all_001 --in_file ./.github/workflows/inputs/in_predict.yaml
sed -i 's/kfold: True/kfold: False/' ./.github/workflows/inputs/in_aegan.yaml

echo "--- Running: AEGAN (Bootstrap) ---"
sed -i 's/bootstrap: False/bootstrap: True/' ./.github/workflows/inputs/in_aegan.yaml
python3 -m xanesnet.cli --mode train_all --in_file ./.github/workflows/inputs/in_aegan.yaml --save
python3 -m xanesnet.cli --mode predict_all --in_model models/aegan_mlp_bootstrap_all_001 --in_file ./.github/workflows/inputs/in_predict.yaml
sed -i 's/bootstrap: True/bootstrap: False/' ./.github/workflows/inputs/in_aegan.yaml

echo "--- Running: AEGAN (Ensemble) ---"
sed -i 's/ensemble: False/ensemble: True/' ./.github/workflows/inputs/in_aegan.yaml
python3 -m xanesnet.cli --mode train_all --in_file ./.github/workflows/inputs/in_aegan.yaml --save
python3 -m xanesnet.cli --mode predict_all --in_model models/aegan_mlp_ensemble_all_001 --in_file ./.github/workflows/inputs/in_predict.yaml
sed -i 's/ensemble: True/ensemble: False/' ./.github/workflows/inputs/in_aegan.yaml

# --- GNN Models ---

echo "--- Running: GNN (Std) ---"
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_gnn.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/gnn_std_xyz_001 --in_file ./.github/workflows/inputs/in_predict_gnn.yaml

echo "--- Running: GNN (Kfold) ---"
sed -i 's/kfold: False/kfold: True/' ./.github/workflows/inputs/in_gnn.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_gnn.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/gnn_kfold_xyz_001 --in_file ./.github/workflows/inputs/in_predict_gnn.yaml
sed -i 's/kfold: True/kfold: False/' ./.github/workflows/inputs/in_gnn.yaml

echo "--- Running: GNN (Bootstrap) ---"
sed -i 's/bootstrap: False/bootstrap: True/' ./.github/workflows/inputs/in_gnn.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_gnn.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/gnn_bootstrap_xyz_001 --in_file ./.github/workflows/inputs/in_predict_gnn.yaml
sed -i 's/bootstrap: True/bootstrap: False/' ./.github/workflows/inputs/in_gnn.yaml

echo "--- Running: GNN (Ensemble) ---"
sed -i 's/ensemble: False/ensemble: True/' ./.github/workflows/inputs/in_gnn.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_gnn.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/gnn_ensemble_xyz_001 --in_file ./.github/workflows/inputs/in_predict_gnn.yaml
sed -i 's/ensemble: True/ensemble: False/' ./.github/workflows/inputs/in_gnn.yaml

echo "--- All tasks completed successfully! ---"