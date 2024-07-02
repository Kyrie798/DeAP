import albumentations as albu
from torchvision import transforms

def get_transforms(size, scope='geometric', crop='random'):
    augs = {'weak': albu.Compose([albu.HorizontalFlip()]),
            'geometric': albu.Compose([albu.HorizontalFlip(),
                                       albu.VerticalFlip(),
                                       albu.RandomRotate90()
                                       ]),
            'None': None
            }

    aug_fn = augs[scope]
    crop_fn = {'random': albu.RandomCrop(size, size, always_apply=True),
               'center': albu.CenterCrop(size, size, always_apply=True)}[crop]

    pipeline = albu.Compose([aug_fn, crop_fn], additional_targets={'target': 'image'})

    def process(a, b):
        r = pipeline(image=a, target=b)
        return r['image'], r['target']

    return process

def get_normalize():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    def process(a, b):
        image = transform(a).permute(1, 2, 0) - 0.5
        target = transform(b).permute(1, 2, 0) - 0.5
        return image, target

    return process