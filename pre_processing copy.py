
import albumentations as albu

def to_tensor_img(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def to_tensor_mask(x, **kwargs):
    return x.astype('float32')

def get_preprocessing():
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """
    _transform = [
        albu.Lambda(image=to_tensor_img, mask=to_tensor_mask),
    ]
    return albu.Compose(_transform)
