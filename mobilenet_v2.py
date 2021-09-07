from torchvision.models.mobilenetv2 import mobilenet_v2

__all__ = ['mobilenet_2', 'mobilenet_1', 'mobilenet_075', 'mobilenet_05', 'mobilenet_025']

def mobilenet_2(num_classes=62):
    model = mobilenet_v2(width_mult=2.0, num_classes=num_classes)
    return model


def mobilenet_1(num_classes=62):
    model = mobilenet_v2(width_mult=1.0, num_classes=num_classes)
    return model


def mobilenet_075(num_classes=62):
    model = mobilenet_v2(width_mult=0.75, num_classes=num_classes)
    return model


def mobilenet_05(num_classes=62):
    model = mobilenet_v2(width_mult=0.5, num_classes=num_classes)
    return model


def mobilenet_025(num_classes=62):
    model = mobilenet_v2(width_mult=0.25, num_classes=num_classes)
    return model