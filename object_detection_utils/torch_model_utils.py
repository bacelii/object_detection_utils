from torchvision.models import resnet
from torchsummary import summary

def enum_extract_value(name):
    if "enum" in str(type(name)):
        name = name.value
    return name

from .backbone_resnet import ResNetBackbones 

def get_resnet_model(name,pretrained = True):
    name = enum_extract_value(
        name = ResNetBackbones.RESNET101             
    )

    if pretrained:
        weights_name = name.replace('r',"R").replace('n',"N")
        weights=f'{weights_name}_Weights.DEFAULT' 
    else:
        weights = None

    model = resnet.__dict__[name](
        weights = weights                                    
    )
    return model

def freeze_parameters(parameter):
    parameter.requires_grad_(False)

def freeze_all_layers_except(
    model,
    layers_to_train=None,
    unfreeze_fc = True,
    verbose = False,
    ):
    """
    Purpose: freeze the parameters of all layers
    except those in list

    """
    if layers_to_train is None:
        layers_to_train = []
    if 'fc' not in layers_to_train and unfreeze_fc:
        layers_to_train.append('fc')

    
    if verbose:
        print(f"layers_to_train = {layers_to_train}")
    for name,parameter in model.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

def layer_names(model):
    return [k[0] for k in list(model.named_parameters())]

def example():
    model = get_resnet_model(ResNetBackbones.RESNET34)
    freeze_all_layers_except(
        model,
        layers_to_train = ['layer4'],
        verbose = True,
    )

    summary(model,(3,288,288))