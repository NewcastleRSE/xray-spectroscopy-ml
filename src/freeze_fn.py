import torch
from torch import nn

def freeze_layers(model_dir, model_mode, freeze_params):
    """Freezes specifed layers in pre-trained model for further training

    Args:
        model_dir (str): Path to pre-trained model
        model_mode (str): Model mode of pre-trained model
        freeze_params (dict): Freeze parameters from input.yaml

    Returns:
        model with frozen layers
    """

    # Load model from file
    model = torch.load(model_dir, map_location=torch.device("cpu"))

    if model_mode == "mlp":
        freeze_layers_mlp(model, freeze_params)
    elif model_mode == "cnn":
        freeze_layers_cnn(model, freeze_params)
    elif model_mode == "ae_mlp" or model_mode == "ae_cnn":
        freeze_layers_ae(model, freeze_params)
    elif model_mode == "aegan_mlp":
        freeze_layers_aegan_mlp(model, freeze_params)
    elif model_mode == "lstm":
        freeze_layers_lstm(model, freeze_params)

    return model


def freeze_layers_mlp(model, freeze_params):
    """
    freeze_params = {
        # Which layers to freeze
        freeze_dense : bool
        # Number of layers to freeze
        n_freeze_dense: int
    }
    """
    freeze_dense = freeze_params["freeze_dense"]

    n_freeze_dense = freeze_params["n_freeze_dense"]

    if freeze_dense:
        count = 0
        for layer in model.dense_layers.children():
            if isinstance(layer, nn.Sequential):
                count += 1
                if count <= n_freeze_dense:
                    for param in layer.parameters():
                        param.requires_grad = False

    return model


def freeze_layers_cnn(model, freeze_params):
    """
    freeze_params = {
        # Which layers to freeze
        freeze_conv : bool
        freeze_dense : bool
        # Number of layers to freeze
        n_freeze_conv : int
        n_freeze_dense : int
    }
    """
    freeze_conv = freeze_params["freeze_conv"]
    freeze_dense = freeze_params["freeze_dense"]

    n_freeze_conv = freeze_params["n_freeze_conv"]
    n_freeze_dense = freeze_params["n_freeze_dense"]

    if freeze_conv:
        count = 0
        for layer in model.conv_layers.children():
            if isinstance(layer, nn.Sequential):
                count += 1
                if count <= n_freeze_conv:
                    for param in layer.parameters():
                        param.requires_grad = False

    if freeze_dense:
        count = 0
        for layer in model.dense_layers.children():
            if isinstance(layer, nn.Sequential):
                count += 1
                if count <= n_freeze_dense:
                    for param in layer.parameters():
                        param.requires_grad = False

    return model


def freeze_layers_ae(model, freeze_params):
    """
    freeze_params = {
        # Which layers to freeze
        freeze_encoder : bool
        freeze_decoder : bool
        freeze_dense : bool
        # Number of layers to freeze
        n_freeze_encoder : int
        n_freeze_decoder : int
        n_freeze_dense : int
    }
    """

    freeze_encoder = freeze_params["freeze_encoder"]
    freeze_decoder = freeze_params["freeze_decoder"]
    freeze_dense = freeze_params["freeze_dense"]

    n_freeze_encoder = freeze_params["n_freeze_encoder"]
    n_freeze_decoder = freeze_params["n_freeze_decoder"]
    n_freeze_dense = freeze_params["n_freeze_dense"]

    if freeze_encoder:
        count = 0
        for layer in model.encoder_layers.children():
            if isinstance(layer, nn.Sequential):
                count += 1
                if count <= n_freeze_encoder:
                    for param in layer.parameters():
                        param.requires_grad = False

    if freeze_decoder:
        count = 0
        for layer in model.decoder_layers.children():
            if isinstance(layer, nn.Sequential):
                count += 1
                if count <= n_freeze_decoder:
                    for param in layer.parameters():
                        param.requires_grad = False
    
    if freeze_dense:
        count = 0
        for layer in model.dense_layers.children():
            if isinstance(layer, nn.Sequential):
                count += 1
                if count <= n_freeze_dense:
                    for param in layer.parameters():
                        param.requires_grad = False
    
    return model



def freeze_layers_aegan_mlp(model, freeze_params):
    """
    freeze_params = {
        # Which layers to freeze
        freeze_encoder1 :
        freeze_encoder2 :
        freeze_decoder1 : bool
        freeze_decoder2 : bool
        freeze_shared_encoder : bool
        freeze_shared_decoder : bool
        freeze_discrim1 : bool
        freeze_discrim2 : bool
        # Number of layers to freeze
        n_freeze_encoder1 : int
        n_freeze_encoder2 : int
        n_freeze_decoder1 : int
        n_freeze_decoder2 : int
        n_freeze_shared_encoder : int
        n_freeze_shared_decoder : int
        n_freeze_discrim1 : int
        n_freeze_discrim2 : int
    }
    """

    freeze_encoder1 = freeze_params["freeze_encoder1"]
    freeze_encoder2 = freeze_params["freeze_encoder2"]
    freeze_decoder1 = freeze_params["freeze_decoder1"]
    freeze_decoder2 = freeze_params["freeze_decoder2"]
    freeze_shared_encoder = freeze_params["freeze_shared_encoder"]
    freeze_shared_decoder = freeze_params["freeze_shared_decoder"]
    freeze_discrim1 = freeze_params["freeze_discrim1"]
    freeze_discrim2 = freeze_params["freeze_discrim2"]

    n_freeze_encoder1 = freeze_params["n_freeze_encoder1"]
    n_freeze_encoder2 = freeze_params["n_freeze_encoder2"]
    n_freeze_decoder1 = freeze_params["n_freeze_decoder1"]
    n_freeze_decoder2 = freeze_params["n_freeze_decoder2"]
    n_freeze_shared_encoder = freeze_params["n_freeze_shared_encoder"]
    n_freeze_shared_decoder = freeze_params["n_freeze_shared_decoder"]
    n_freeze_discrim1 = freeze_params["n_freeze_discrim1"]
    n_freeze_discrim2 = freeze_params["n_freeze_discrim2"]

    # Encoder 1
    if freeze_encoder1:
        count = 0
        for layer in model.gen_a.encoder_layers.children():
            if isinstance(layer, nn.Sequential):
                count += 1
                if count <= n_freeze_encoder1:
                    for param in layer.parameters():
                        param.requires_grad = False

    # Encoder 2
    if freeze_encoder2:
        count = 0
        for layer in model.gen_b.encoder_layers.children():
            if isinstance(layer, nn.Sequential):
                count += 1
                if count <= n_freeze_encoder2:
                    for param in layer.parameters():
                        param.requires_grad = False

    # Decoder 1
    if freeze_decoder1:
        count = 0
        for layer in model.gen_a.decoder_layers.children():
            if isinstance(layer, nn.Sequential):
                count += 1
                if count <= n_freeze_decoder1:
                    for param in layer.parameters():
                        param.requires_grad = False

    # Decoder 2
    if freeze_decoder2:
        count = 0
        for layer in model.gen_b.decoder_layers.children():
            if isinstance(layer, nn.Sequential):
                count += 1
                if count <= n_freeze_decoder2:
                    for param in layer.parameters():
                        param.requires_grad = False


    # Shared Encoder Layers
    if freeze_shared_encoder:
        count = 0
        for layer in model.enc_shared.dense_layers.children():
            if isinstance(layer, nn.Sequential):
                count += 1
                if count <= n_freeze_shared_encoder:
                    for param in layer.parameters():
                        param.requires_grad = False

    if freeze_shared_decoder:
        count = 0
        for layer in model.dec_shared.dense_layers.children():
            if isinstance(layer, nn.Sequential):
                count += 1
                if count <= n_freeze_shared_decoder:
                    for param in layer.parameters():
                        param.requires_grad = False

    # Discriminator 1
    if freeze_discrim1:
        count = 0
        for layer in model.dis_a.dense_layers.children():
            if isinstance(layer, nn.Sequential):
                count += 1
                if count <= n_freeze_discrim1:
                    for param in layer.parameters():
                        param.requires_grad = False

    # Discriminator 2
    if freeze_discrim2:
        count = 0
        for layer in model.dis_b.dense_layers.children():
            if isinstance(layer, nn.Sequential):
                count += 1
                if count <= n_freeze_discrim2:
                    for param in layer.parameters():
                        param.requires_grad = False

    return model


def freeze_layers_lstm(model, freeze_params):
    """
    freeze_params = {
        # Which layers to freeze
        freeze_lstm: bool
        freeze_dense: bool
        # Number of layers to freeze
        n_freeze_lstm: int
        n_freeze_dense: int
    }
    """
    freeze_lstm = freeze_params["freeze_lstm"]
    freeze_dense = freeze_params["freeze_dense"]

    n_freeze_lstm = freeze_params["n_freeze_lstm"]
    n_freeze_dense = freeze_params["n_freeze_dense"]

    if freeze_lstm:
        LSTM_BLOCK_SIZE = 8 
        count = 0
        for name, param in model.lstm.named_parameters():
            if count // LSTM_BLOCK_SIZE + 1 <= n_freeze_lstm:
                param.requires_grad = False
            count += 1

    if freeze_dense:
        print(">>> Note current implementation freezes all model.fc layers.")
        count = 0
        for name, param in model.fc.named_parameters():
            param.requires_grad = False

    return model