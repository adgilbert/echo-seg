from .net import get_network


def create_model(opt):
    if type(opt) != dict:
        return get_network(**vars(opt))
    return get_network(**opt)
