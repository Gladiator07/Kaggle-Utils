import copy
import os

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from transformers import ProgressCallback


class CustomProgressCallback(ProgressCallback):
    def __init__(self):
        super().__init__()

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if state.is_local_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None
        epoch = metrics.pop("epoch", None)
        summary = " | ".join([f"{m}: {v:.4f}" for m, v in metrics.items()])
        summary_str = f"\nEpoch {epoch}  | {summary}\n"
        self.training_bar.write(summary_str)

    def on_log(self, args, state, control, logs=None, **kwargs):
        pass
        # if state.is_local_process_zero and self.training_bar is not None:
        #     _ = logs.pop("total_flos", None)
        #     self.training_bar.write(str(logs))


def asHours(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:.0f}h:{m:.0f}m:{s:.0f}s"


def print_line():
    prefix, unit, suffix = "\n#", "--", "#\n"
    print(prefix + unit * 50 + suffix)


# ----------------------------------------- Util Functions -----------------------------------------
def process_config_for_wandb(cfg: OmegaConf):
    tmp_cfg = copy.deepcopy(cfg)
    cfg_dict = OmegaConf.to_container(tmp_cfg, resolve=True, throw_on_missing=True)
    del cfg_dict["trainer_args"]
    return cfg_dict


# ---------------------------------- Custom Metrics func ---------------------------------- #
def compute_metrics(p):
    # as model output can return multiple things, take first from the tuple for logits
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids
    col_rmse = np.sqrt(np.mean((logits - labels) ** 2, axis=0))
    mcrmse = np.mean(col_rmse)

    return {
        "content_rmse": round(float(col_rmse[0]), 4),
        "wording_rmse": round(float(col_rmse[1]), 4),
        "mcrmse": round(float(mcrmse), 4),
    }


def save_oof_predictions(
    oof_df: pd.DataFrame, logits: np.ndarray, labels: np.ndarray, out_dir: str
):
    oof_df["pred_content"] = logits[:, 0]
    oof_df["pred_wording"] = logits[:, 1]
    # calculate metric for each sample for analyzing where the model fails
    content_rmse_per_sample = np.sqrt((logits[:, 0] - labels[:, 0]) ** 2)
    wording_rmse_per_sample = np.sqrt((logits[:, 1] - labels[:, 1]) ** 2)
    col_rmse = (content_rmse_per_sample + wording_rmse_per_sample) / 2
    oof_df["content_mrmse"] = content_rmse_per_sample
    oof_df["wording_mrmse"] = wording_rmse_per_sample
    oof_df["rmse"] = col_rmse
    oof_df.to_csv(os.path.join(out_dir, "oof.csv"), index=False)
    return oof_df
