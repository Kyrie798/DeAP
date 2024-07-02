import setproctitle
from train_config import config as cfg
from lib.core.base_trainer.net_work import Train

setproctitle.setproctitle('DeAP')

def main():
    trainer = Train(cfg=cfg)
    trainer.loop()

if __name__ == '__main__':
    main()