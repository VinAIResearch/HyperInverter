from PIL import Image


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def get_keys(d, name, key="state_dict"):
    if key in d:
        d = d[key]
    d_filt = {k[len(name) + 1 :]: v for k, v in d.items() if k[: len(name)] == name}
    return d_filt


def convert_predicted_weights_to_dict(pred_weights_per_sample):
    """
    Convert data like "conv1.affine.weight : value" to
    {
            "conv1": {
                    "affine": {
                            "weight": value
                    }
            }
            "torgb" : {
                    ...
            }
            ...
    }
    #
    """
    added_weights = {}
    for key in pred_weights_per_sample:
        cur = added_weights
        attr_names = key.split(".")
        for i, attr_name in enumerate(attr_names):
            if i == len(attr_names) - 1:
                cur[attr_name] = pred_weights_per_sample[key]
            elif attr_name not in cur:
                cur[attr_name] = {}
            cur = cur[attr_name]
    return added_weights


# Log images
def log_input_image(x):
    return tensor2im(x)


def tensor2im(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = (var + 1) / 2
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype("uint8"))
