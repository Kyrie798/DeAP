from easydict import EasyDict as edict
from lib.utils.seed_utils import seed_everything

config = edict()
config.TRAIN = edict()
config.TRAIN.DDP = False
config.TRAIN.batch_size = 4
config.TRAIN.process_num = 4
config.TRAIN.lr = 1.e-4
config.TRAIN.min_lr = 1.e-7
config.TRAIN.epochs = 3000
config.TRAIN.MCFM_epoch = 0

config.TEST = edict()
config.TEST.blur = './datasets/GoPro/test/blur/'
config.TEST.restored = './out'

config.MODEL = edict()
config.MODEL.model_path = 'checkpoint'

config.DATA = edict()
config.DATA.train_blur = './datasets/GoPro/train/blur/**/*.png'
config.DATA.train_sharp = './datasets/GoPro/train/sharp/**/*.png'
config.DATA.val_blur = './datasets/GoPro/test/blur/**/*.png'
config.DATA.val_sharp = './datasets/GoPro/test/sharp/**/*.png'
config.DATA.Patch_size = 512

config.SEED = 10086
seed_everything(config.SEED)