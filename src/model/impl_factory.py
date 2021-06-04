from model.autoenc_deconv import Autoenc_deconv
from model.autoenc_light import Autoenc_light

def model_impl_factory(cfg: dict, **kwargs):
    """Returns model implementation based on config
    """
    if cfg['model'].get('name', None) is None or cfg['model']['name'] == 'autoenc_deconv':
        return Autoenc_deconv(kwargs['ds_mean'], params=cfg['model'])
    if cfg['model']['name'] == 'autoenc_light':
        return Autoenc_light(kwargs['ds_mean'], params=cfg['model'])