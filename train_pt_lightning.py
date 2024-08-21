import os
from pprint import pprint

import math
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
from pytorch_lightning.tuner import Tuner

from libs.datasets.avdf1m import AVDF1MDataModule
from libs.modeling.full_model_pytorch_lightnining import IdentityPtTransformer
from libs.core import load_config

import fire

def main(
  config_file: str,
  data_root_folder: str,
  n_gpus: int = 1,
  epochs: int = 50,
  gradient_accumulation_steps: int = 4,
  resume_checkpoint: str|None = None,      
)->None:
    
    if os.path.isfile(config_file):
        cfg = load_config(config_file)
    else:
        raise ValueError("Config file does not exist.")
    
    pprint(cfg)

    dm =  AVDF1MDataModule(
        train_split=cfg['train_split'],
        val_split=cfg['val_split'],
        test_split=cfg['test_split'],
        root=data_root_folder,
        train_json=os.path.join(data_root_folder, cfg['dataset']['train_json_file']),
        val_json=os.path.join(data_root_folder, cfg['dataset']['val_json_file']),
        test_json=os.path.join(data_root_folder, cfg['dataset']['test_json_file']),
        feat_stride=cfg['dataset']['feat_stride'],
        num_frames=cfg['dataset']['num_frames'],
        max_seq_len=cfg['dataset']['max_seq_len'],
        trunc_thres=cfg['dataset']['trunc_thresh'],
        num_classes=cfg['dataset']['num_classes'],
        force_upsampling=cfg['dataset']['force_upsampling'],
        default_fps=cfg['dataset']['default_fps'],
        img_size=cfg['dataset']['img_size'],
        batch_size=cfg['loader']['batch_size'],
        num_workers=cfg['loader']['num_workers']
    )
    
    dm.setup()
    val_metadata = dm.val_dataset.data_list
    # print(len(val_metadata))
    # val1 = dm.val_dataset.__getitem__(0)
    # print(val1['video_id'], val1['video'].shape, val1['audio'].shape) 
    # print(val1['segments'])
    # print(val1['labels'], val1['av_labels'])
    # print(val1['fps'], val1['duration'], val1['actual_frames'], val1['feat_stride'], val1['feat_num_frames'])

    model = IdentityPtTransformer(
        model_name=cfg['model_name'],
        optimizer_config=cfg['opt'],
        num_iters_per_epoch=len(dm.train_dataloader()),
        distributed=n_gpus>1,
        val_metadata=val_metadata,
        **cfg['model']
    )

    monitor = "val_MAP"

    print('setting up trainer')
    trainer = Trainer(log_every_n_steps=50, precision="16-mixed", gradient_clip_val=0.5,
                      max_epochs=epochs, accumulate_grad_batches=gradient_accumulation_steps,
                      callbacks=[
                          ModelCheckpoint( dirpath=f"./ckpt", save_last=True, filename=f'identity_actionformer' + "-{epoch}-{val_loss:.3f}", monitor=monitor, mode="min"),
                          EarlyStopping(monitor=monitor, mode='min', verbose=False, patience=7),
                          StochasticWeightAveraging(1e-2)
                          ], 
                      enable_checkpointing=True,
                      benchmark=True,
                      accelerator="gpu",
                      devices=n_gpus,
                      strategy="auto" if n_gpus < 2 else "ddp"
                    )

    print('trainer set')
    if resume_checkpoint is None:
        print('starting training')
        # tuner = Tuner(trainer)
        # tuner.lr_find(model,datamodule=dm)
        # print(model.learning_rate)
        trainer.fit(model, dm)
    else:
        print('resuming traning')
        trainer.fit(model=model, datamodule=dm, ckpt_path=resume_checkpoint)

if __name__ == "__main__":
    fire.Fire(main)


# python train_pt_lightning.py --config_file /home/users/ntu/ashutosh/scratch/Codes/avdf_actionformer_baseline/configs/avdf1m_config.yaml --data_root_folder /home/project/12001458/1MDeepfake_challenge/ --n_gpus 1