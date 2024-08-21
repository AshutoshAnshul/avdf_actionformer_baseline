from pytorch_lightning import LightningModule
from torch import Tensor, stack, tensor
from typing import Sequence, Optional, Union, Tuple

from libs.modeling.models import make_meta_arch
from libs.utils.train_utils import make_optimizer, make_scheduler, ModelEma
from libs.utils.metrics import AP, AR

from statistics import mean


class IdentityPtTransformer(LightningModule):
    def __init__(self,
        model_name, optimizer_config, num_iters_per_epoch, distributed, val_metadata, **kwargs
    ):
        super().__init__()
        self.model = make_meta_arch(model_name, **kwargs)
        self.optimizer_config = optimizer_config
        self.num_iters_per_epoch = num_iters_per_epoch
        self.model_ema = ModelEma(self.model)
        
        self.distributed = distributed

        self.val_metadata = val_metadata
        self.APIOUs = [0.5, 0.75, 0.95]
        self.APevaluator = AP(iou_thresholds=self.APIOUs)
        self.ARIOUs = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.ARProposals = [100, 50, 20, 10]
        self.ARevaluator = AR(iou_thresholds=self.ARIOUs, n_proposals_list=self.ARProposals)
        self.epoch_result = {}

    def forward(self, batch: Sequence[Tensor]) -> Sequence[Tensor]:

        output = self.model(batch)

        return output
    
    
    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None
    ) -> Tensor:
        loss_dict = self.forward(batch)
        self.model_ema.update(self.model)

        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=self.distributed)
        
        return loss_dict["final_loss"]
    
    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None
    ) -> None:
        output = self.forward(batch)

        num_vids = len(output)
        for vid_idx in range(num_vids):
            vid_id = output[vid_idx]['video_id']
            starts = tensor(output[vid_idx]['segments'][:, 0])
            ends = tensor(output[vid_idx]['segments'][:, 1])
            scores = tensor(output[vid_idx]['scores'])

            proposal_list = stack((scores, starts, ends), dim=1)

            self.epoch_result[vid_id] = proposal_list

    def on_validation_epoch_end(self):

        AP_score = self.APevaluator(self.val_metadata, self.epoch_result)
        AR_score = self.ARevaluator(self.val_metadata, self.epoch_result)
        
        log_dict = {}
        for iou_thres in self.APIOUs:
            log_dict[f"val_AP@{iou_thres}"] = AP_score[iou_thres]

        for n_prop in self.ARProposals:
            log_dict[f"val_AR@{n_prop}"] = AR_score[n_prop]

        MAP_Val = mean([AP_score[k] for k in AP_score])
        log_dict["val_MAP"] = MAP_Val

        self.log_dict(log_dict, on_epoch=True, sync_dist=self.distributed)

        self.epoch_result.clear()
        
        return MAP_Val

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Tuple [Tensor, Tensor, Tensor, Tensor, Tensor]:
        
        output = self.forward(batch)

        return output
    
    def configure_optimizers(self):
        optimizer = make_optimizer(self.model, self.optimizer_config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": make_scheduler(optimizer, self.optimizer_config, self.num_iters_per_epoch),
                "monitor": "val_MAP"
            }
        }

 