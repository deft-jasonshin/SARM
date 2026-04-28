import os
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from tqdm import tqdm
import wandb

from lerobot.common.datasets.rm_lerobot_dataset import FrameGapLeRobotDataset 
from utils.data_utils import get_valid_episodes, split_train_eval_episodes, adapt_lerobot_batch_sarm
from utils.train_utils import set_seed, save_ckpt, get_normalizer_from_calculated, plot_episode_result_raw_data, plot_episode_result
from utils.raw_data_utils import get_frame_num, get_frame_data_fast, get_traj_data, normalize_sparse, normalize_dense
from models.subtask_estimator import SubtaskTransformer
from models.stage_estimator import StageTransformer
from models.clip_encoder import FrozenCLIPEncoder
from utils.make_demo_video import produce_video
from utils.pred_smoother import RegressionConfidenceSmoother

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_IGNORE_GLOBS"] = "**/rollout/**"
# os.environ["WANDB_MODE"] = "disabled"

def infinite_loader(dl):
    """Yield batches forever; reshuffles each pass if dl.shuffle=True."""
    while True:
        for b in dl:
            yield b

            
class SARMWorkspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.general.device if torch.cuda.is_available() else "cpu")
        print(f"[Init] Using device: {self.device}")
        set_seed(cfg.general.seed)
        self.camera_names = cfg.general.camera_names
        self.save_dir = Path(f'{cfg.general.project_name}/{cfg.general.task_name}')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Init] Logging & ckpts to: {self.save_dir}")

    def gen_stage_emb(self, num_classes, trg):
        """
        Returns stage_onehot with a modality dim (B, 1, T, C).
        """
        # integer part of float targets -> [0, C-1]
        idx = trg.long().clamp(min=0, max=num_classes - 1)   # (B, T)

        C = num_classes
        # identity-lookup one-hot
        stage_onehot = torch.eye(C, device=trg.device)[idx]            # (B, T, C)
        stage_onehot = stage_onehot.unsqueeze(1)                       # (B, 1, T, C)
        return stage_onehot
    
    def train(self):
        cfg = self.cfg
        OmegaConf.save(cfg, self.save_dir / "config.yaml")
        # --- wandb ---
        wandb.init(
            project=f'{cfg.general.project_name}-{cfg.general.task_name}',
            name=f'{datetime.now().strftime("%Y.%m.%d-%H.%M.%S")}',
            config=cfg,
        )

        # --- data ---
        train_mode = getattr(cfg.train, "train_mode", "both")
        use_sparse = train_mode in ("sparse", "both")
        use_dense = train_mode in ("dense", "both")

        if use_sparse:
            valid_episodes_sparse = get_valid_episodes(cfg.general.repo_id_sparse)
            train_eps_sparse, val_eps_sparse = split_train_eval_episodes(valid_episodes_sparse, 1 - cfg.train.val_portion, seed=cfg.general.seed)
            dataset_train_sparse = FrameGapLeRobotDataset(repo_id=cfg.general.repo_id_sparse, 
                                                   root=cfg.general.repo_id_sparse,
                                                   episodes=train_eps_sparse, 
                                                   n_obs_steps=cfg.model.n_obs_steps, 
                                                   frame_gap=cfg.model.frame_gap,
                                                   max_rewind_steps=cfg.model.max_rewind_steps,
                                                   image_names=cfg.general.camera_names,
                                                   annotation_list=cfg.model.sparse_annotation_list,
                                                   task_name=cfg.general.task_name)
            dataset_val_sparse = FrameGapLeRobotDataset(repo_id=cfg.general.repo_id_sparse, 
                                                   root=cfg.general.repo_id_sparse,
                                                   episodes=val_eps_sparse, 
                                                   n_obs_steps=cfg.model.n_obs_steps, 
                                                   frame_gap=cfg.model.frame_gap,
                                                   max_rewind_steps=cfg.model.max_rewind_steps,
                                                   image_names=cfg.general.camera_names,
                                                   annotation_list=cfg.model.sparse_annotation_list,
                                                   task_name=cfg.general.task_name)
            dataloader_train_sparse = torch.utils.data.DataLoader(dataset_train_sparse, **cfg.dataloader)
            dataloader_val_sparse   = torch.utils.data.DataLoader(dataset_val_sparse, **cfg.val_dataloader)

        if use_dense:
            valid_episodes_dense = get_valid_episodes(cfg.general.repo_id_dense)
            train_eps_dense, val_eps_dense = split_train_eval_episodes(valid_episodes_dense, 1 - cfg.train.val_portion, seed=cfg.general.seed)
            dataset_train_dense = FrameGapLeRobotDataset(repo_id=cfg.general.repo_id_dense, 
                                                   root=cfg.general.repo_id_dense,
                                                   episodes=train_eps_dense, 
                                                   n_obs_steps=cfg.model.n_obs_steps, 
                                                   frame_gap=cfg.model.frame_gap,
                                                   max_rewind_steps=cfg.model.max_rewind_steps,
                                                   image_names=cfg.general.camera_names,
                                                   annotation_list=cfg.model.dense_annotation_list,
                                                   task_name=cfg.general.task_name)
            dataset_val_dense = FrameGapLeRobotDataset(repo_id=cfg.general.repo_id_dense, 
                                                   root=cfg.general.repo_id_dense,
                                                   episodes=val_eps_dense, 
                                                   n_obs_steps=cfg.model.n_obs_steps, 
                                                   frame_gap=cfg.model.frame_gap,
                                                   max_rewind_steps=cfg.model.max_rewind_steps,
                                                   image_names=cfg.general.camera_names,
                                                   annotation_list=cfg.model.dense_annotation_list,
                                                   task_name=cfg.general.task_name)
            dataloader_train_dense = torch.utils.data.DataLoader(dataset_train_dense, **cfg.dataloader)
            dataloader_val_dense   = torch.utils.data.DataLoader(dataset_val_dense, **cfg.val_dataloader)
        state_normalizer = get_normalizer_from_calculated(cfg.general.state_norm_path, self.device)

        # --- encoders ---
        # CLIP
        clip_encoder = FrozenCLIPEncoder(cfg.encoders.vision_ckpt, self.device)
        vis_dim = 512
        txt_dim = 512


        # --- reward_model ---
        subtask_model = SubtaskTransformer(d_model=cfg.model.d_model, 
                                  vis_emb_dim=vis_dim, 
                                  text_emb_dim=txt_dim,
                                  state_dim=cfg.model.state_dim,
                                  n_layers=cfg.model.n_layers,
                                  n_heads=cfg.model.n_heads,
                                  dropout=cfg.model.dropout,
                                  num_cameras=len(self.camera_names),
                                  ).to(self.device)
        stage_model = StageTransformer(d_model=cfg.model.d_model, 
                                  vis_emb_dim=vis_dim, 
                                  text_emb_dim=txt_dim,
                                  state_dim=cfg.model.state_dim,
                                  n_layers=cfg.model.n_layers,
                                  n_heads=cfg.model.n_heads,
                                  dropout=cfg.model.dropout,
                                  num_cameras=len(self.camera_names),
                                  num_classes_sparse=cfg.model.num_classes_sparse,
                                  num_classes_dense=cfg.model.num_classes_dense
                                  ).to(self.device)
        
        if cfg.model.resume_training:
            reward_model_path = Path(cfg.model.model_path)
            # Load checkpoints
            subtask_model_path = reward_model_path / "subtask_best.pt"; stage_model_path = reward_model_path / "stage_best.pt"
            subtask_ckpt = torch.load(subtask_model_path, map_location=self.device); stage_ckpt = torch.load(stage_model_path, map_location=self.device)
            subtask_model.load_state_dict(subtask_ckpt["model"]); stage_model.load_state_dict(stage_ckpt["model"])
            subtask_model.to(self.device); stage_model.to(self.device)
            subtask_model.train(); stage_model.train()


        # Optimizer
        subtask_optimizer = torch.optim.AdamW(
            subtask_model.parameters(),
            lr=cfg.optim.lr,
            betas=tuple(cfg.optim.betas),
            eps=cfg.optim.eps,
            weight_decay=cfg.optim.weight_decay,
        )
        stage_optimizer = torch.optim.AdamW(
            stage_model.parameters(),
            lr=cfg.optim.lr,
            betas=tuple(cfg.optim.betas),
            eps=cfg.optim.eps,
            weight_decay=cfg.optim.weight_decay,
        )


        # Schedulers
        # subtask scheduler
        subtask_warmup_scheduler = LinearLR(
            subtask_optimizer,
            start_factor=1e-6 / cfg.optim.lr,  # or 0.0 for full ramp-up
            end_factor=1.0,
            total_iters=cfg.optim.warmup_steps
        )
        subtask_cosine_scheduler = CosineAnnealingLR(
            subtask_optimizer,
            T_max=cfg.optim.total_steps - cfg.optim.warmup_steps,  # cosine decay after warmup
            eta_min=0.0  # or set a nonzero final LR if needed
        )
        subtask_scheduler = SequentialLR(
            subtask_optimizer,
            schedulers=[subtask_warmup_scheduler, subtask_cosine_scheduler],
            milestones=[cfg.optim.warmup_steps]
        )

        # Stage scheduler
        stage_warmup_scheduler = LinearLR(
            stage_optimizer,
            start_factor=1e-6 / cfg.optim.lr,  # can be 0.0 if you prefer
            end_factor=1.0,
            total_iters=cfg.optim.warmup_steps
        )
        stage_cosine_scheduler = CosineAnnealingLR(
            stage_optimizer,
            T_max=cfg.optim.total_steps - cfg.optim.warmup_steps,
            eta_min=0.0
        )
        stage_scheduler = SequentialLR(
            stage_optimizer,
            schedulers=[stage_warmup_scheduler, stage_cosine_scheduler],
            milestones=[cfg.optim.warmup_steps]
        )

        def train_step(batch, anno_type):
            B, T = batch["image_frames"][self.camera_names[0]].shape[:2]
            img_list = []
            for key in self.camera_names:
                imgs = batch["image_frames"][key].flatten(0, 1).to(self.device) # (B*T, C, H, W)
                img_list.append(imgs)
            
            lang_strs = batch["tasks"]
            trg = batch["targets"].to(self.device)
            lens = batch["lengths"].to(self.device)
            state = batch["state"].to(self.device)
            gt_stage, gt_sub_reward = torch.floor(trg).to(torch.long), torch.remainder(trg, 1.0)

            with torch.no_grad():
                state = state_normalizer.normalize(state)
                imgs_all = torch.cat(img_list, dim=0)  # (N * B * T, C, H, W)
                img_emb = clip_encoder.encode_image(imgs_all)  # (N * B * T, D)
                img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)  # (B, N, T, D)
                lang_emb = clip_encoder.encode_text(lang_strs) # lang_emb: (B, txt_dim)

            if cfg.model.no_state:
                state = torch.zeros_like(state, device=self.device)
            stage_pred = stage_model(img_emb, lang_emb, state, lens, scheme=anno_type)  # (B, N, T, num_classes)
            
            # Inject stage prior to subtask model
            if anno_type == "sparse":
                num_classes = cfg.model.num_classes_sparse
            else:
                num_classes = cfg.model.num_classes_dense
            
            if torch.rand(1).item() < 0.5:
                # Mode 1: ground truth trg -> one-hot
                stage_emb = self.gen_stage_emb(num_classes, trg)                              # (B, 1, T, C)
                subtask_pred = subtask_model(img_emb, lang_emb, state, lens, stage_emb)
            else:
                # Mode 2: predicted argmax -> one-hot
                stage_idx = stage_pred.argmax(dim=-1)                                         # (B, T)
                stage_onehot = F.one_hot(stage_idx, num_classes=stage_pred.size(-1)).float()  # (B, T, C)
                stage_emb = stage_onehot.unsqueeze(1)                                         # (B, 1, T, C)
                subtask_pred = subtask_model(img_emb, lang_emb, state, lens, stage_emb)
            
            if anno_type == "sparse":
                stage_loss = F.cross_entropy(stage_pred.view(-1, num_classes), gt_stage.view(-1), reduction="mean")
            else: 
                stage_loss = F.cross_entropy(stage_pred.view(-1, num_classes), gt_stage.view(-1), reduction="mean")
            subtask_loss = F.mse_loss(subtask_pred, gt_sub_reward, reduction="mean")

            subtask_optimizer.zero_grad()
            subtask_loss.backward()
            subtask_unclipped = nn.utils.clip_grad_norm_(subtask_model.parameters(), float("inf")).item()
            _ = nn.utils.clip_grad_norm_(subtask_model.parameters(), cfg.train.grad_clip)
            subtask_optimizer.step()
            subtask_scheduler.step()

            stage_optimizer.zero_grad()
            stage_loss.backward()
            stage_unclipped = nn.utils.clip_grad_norm_(stage_model.parameters(), float("inf")).item()
            _ = nn.utils.clip_grad_norm_(stage_model.parameters(), cfg.train.grad_clip)
            stage_optimizer.step()
            stage_scheduler.step()
            
            return {
                    "train/stage_loss": stage_loss.item(),
                    "train/subtask_loss": subtask_loss.item(),
                    "train/total_loss": (stage_loss.item() + subtask_loss.item()),
                    "train/lr": subtask_scheduler.get_last_lr()[0],
                    "train/subtask_grad_norm": subtask_unclipped,
                    "train/stage_grad_norm": stage_unclipped,
                }

        with torch.no_grad():
            def valid_step(batch, anno_type):
                B, T = batch["image_frames"][self.camera_names[0]].shape[:2]
                img_list = []
                for key in self.camera_names:
                    imgs = batch["image_frames"][key].flatten(0, 1).to(self.device) # (B*T, C, H, W)
                    img_list.append(imgs)
                
                lang_strs = batch["tasks"]
                trg = batch["targets"].to(self.device)
                lens = batch["lengths"].to(self.device)
                state = batch["state"].to(self.device)
                gt_stage, gt_sub_reward = torch.floor(trg).to(torch.long), torch.remainder(trg, 1.0)
                state = state_normalizer.normalize(state)

                # VLM encoding
                imgs_all = torch.cat(img_list, dim=0)  # (N * B * T, C, H, W)
                img_emb = clip_encoder.encode_image(imgs_all)  # (N * B * T, D)
                img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)  # (B, N, T, D)
                lang_emb = clip_encoder.encode_text(lang_strs) # lang_emb: (B, txt_dim)

                if cfg.model.no_state:
                    state = torch.zeros_like(state, device=self.device)
                stage_pred = stage_model(img_emb, lang_emb, state, lens, scheme=anno_type)  # (B, N, T, num_classes)
                
                # Inject stage prior to subtask model
                stage_idx = stage_pred.argmax(dim=-1)                          # (B, T)
                stage_onehot = F.one_hot(stage_idx, num_classes=stage_pred.size(-1)).float()  # (B, T, C)
                stage_emb = stage_onehot.unsqueeze(1)                          # (B, 1, T, C)
                subtask_pred = subtask_model(img_emb, lang_emb, state, lens, stage_emb)
                
                if anno_type == "sparse":
                    stage_loss = F.cross_entropy(stage_pred.view(-1, cfg.model.num_classes_sparse), gt_stage.view(-1), reduction="mean")
                else: 
                    stage_loss = F.cross_entropy(stage_pred.view(-1, cfg.model.num_classes_dense), gt_stage.view(-1), reduction="mean")
                subtask_loss = F.mse_loss(subtask_pred, gt_sub_reward, reduction="mean")

                return {
                        "train/stage_loss": stage_loss.item(),
                        "train/subtask_loss": subtask_loss.item(),
                        "train/total_loss": (stage_loss.item() + subtask_loss.item()),
                        "train/lr": subtask_scheduler.get_last_lr()[0],
                    }

        if use_dense:
            dense_iter_train = infinite_loader(dataloader_train_dense)
            dense_iter_val = infinite_loader(dataloader_val_dense)

        if use_sparse:
            primary_train_loader = dataloader_train_sparse
            primary_val_loader = dataloader_val_sparse
        else:
            primary_train_loader = dataloader_train_dense
            primary_val_loader = dataloader_val_dense
        
        # ==================== training loop ==================================
        best_val = float("inf")
        step = 0
        
        for epoch in range(1, cfg.train.num_epochs + 1):
            subtask_model.train(); stage_model.train()
            with tqdm(primary_train_loader, desc=f"Epoch {epoch}") as pbar:
                for primary_batch in pbar:
                    if use_sparse:
                        sparse_batch = adapt_lerobot_batch_sarm(primary_batch, camera_names=cfg.general.camera_names)
                        sparse_result = train_step(sparse_batch, anno_type="sparse")
                    if use_dense:
                        if train_mode == "both":
                            dense_batch = adapt_lerobot_batch_sarm(next(dense_iter_train), camera_names=cfg.general.camera_names)
                        else:
                            dense_batch = adapt_lerobot_batch_sarm(primary_batch, camera_names=cfg.general.camera_names)
                        dense_result = train_step(dense_batch, anno_type="dense")

                    if step % cfg.train.log_every == 0:
                        if use_sparse:
                            wandb.log({f"sparse/{k}": v for k, v in sparse_result.items()}, step=step)
                        if use_dense:
                            wandb.log({f"dense/{k}": v for k, v in dense_result.items()}, step=step)

                    if train_mode == "both":
                        stage_loss = (sparse_result["train/stage_loss"] + dense_result["train/stage_loss"])/2
                        subtask_loss = (sparse_result["train/subtask_loss"] + dense_result["train/subtask_loss"])/2
                    elif train_mode == "sparse":
                        stage_loss = sparse_result["train/stage_loss"]
                        subtask_loss = sparse_result["train/subtask_loss"]
                    else:
                        stage_loss = dense_result["train/stage_loss"]
                        subtask_loss = dense_result["train/subtask_loss"]
                    pbar.set_postfix(loss=f"{(stage_loss + subtask_loss):.4f}")

                    if step % cfg.train.save_every == 0:
                        save_ckpt(subtask_model, subtask_optimizer, epoch, self.save_dir, input_name=f"subtask_step_{step:06d}_loss_{subtask_loss:.3f}")
                        save_ckpt(stage_model, stage_optimizer, epoch, self.save_dir, input_name=f"stage_step_{step:06d}_loss_{stage_loss:.3f}")

                    step += 1

            # --- validation ---
            if epoch % cfg.train.eval_every == 0:
                subtask_model.eval(); stage_model.eval()
                total_loss, num = 0.0, 0
                print("running validation...")
                with torch.no_grad():
                    for primary_batch in primary_val_loader:
                        if use_sparse:
                            sparse_batch = adapt_lerobot_batch_sarm(primary_batch, camera_names=cfg.general.camera_names)
                            sparse_result = valid_step(sparse_batch, anno_type="sparse")
                        if use_dense:
                            if train_mode == "both":
                                dense_batch = adapt_lerobot_batch_sarm(next(dense_iter_val), camera_names=cfg.general.camera_names)
                            else:
                                dense_batch = adapt_lerobot_batch_sarm(primary_batch, camera_names=cfg.general.camera_names)
                            dense_result = valid_step(dense_batch, anno_type="dense")

                        if step % cfg.train.log_every == 0:
                            if use_sparse:
                                wandb.log({f"sparse/{k}": v for k, v in sparse_result.items()}, step=step)
                            if use_dense:
                                wandb.log({f"dense/{k}": v for k, v in dense_result.items()}, step=step)

                        if train_mode == "both":
                            stage_loss = (sparse_result["train/stage_loss"] + dense_result["train/stage_loss"])/2
                            subtask_loss = (sparse_result["train/subtask_loss"] + dense_result["train/subtask_loss"])/2
                        elif train_mode == "sparse":
                            stage_loss = sparse_result["train/stage_loss"]
                            subtask_loss = sparse_result["train/subtask_loss"]
                        else:
                            stage_loss = dense_result["train/stage_loss"]
                            subtask_loss = dense_result["train/subtask_loss"]

                        total_loss += (stage_loss + subtask_loss)
                        num += 1

                val_loss = total_loss / num 
                print(f"[Eval] Epoch {epoch} Val L1: {val_loss:.6f}")
                wandb.log({"val/loss": val_loss}, step=step)

            torch.cuda.empty_cache()


            # --- save checkpoints ---
            save_ckpt(subtask_model, subtask_optimizer, epoch, self.save_dir, input_name="subtask_latest")
            save_ckpt(stage_model, stage_optimizer, epoch, self.save_dir, input_name="stage_latest")
            
            if epoch == cfg.train.num_epochs:
                save_ckpt(subtask_model, subtask_optimizer, epoch, self.save_dir, input_name="subtask_final")
                save_ckpt(stage_model, stage_optimizer, epoch, self.save_dir, input_name="stage_final")
            
            if val_loss < best_val:
                best_val = val_loss
                save_ckpt(subtask_model, subtask_optimizer, epoch, self.save_dir, input_name="subtask_best")
                save_ckpt(stage_model, stage_optimizer, epoch, self.save_dir, input_name="stage_best")

        print(f"Training done. Best val_loss MSE = {best_val}")
        wandb.finish()



    # Evaluate whole trajectory from demo data, generating video
    def eval(self):
        import random
        cfg = self.cfg
        model_type = cfg.eval.model_type
        dataset_type = cfg.eval.dataset_type
        if dataset_type == "sparse":
            repo_id = cfg.general.repo_id_sparse
        else:
            repo_id = cfg.general.repo_id_dense
        
        
        valid_episodes = get_valid_episodes(repo_id)
        train_eps, val_eps = split_train_eval_episodes(valid_episodes, 1 - cfg.train.val_portion, seed=cfg.general.seed)
        dataset_val = FrameGapLeRobotDataset(repo_id=repo_id, 
                                               episodes=val_eps, 
                                               n_obs_steps=cfg.model.n_obs_steps, 
                                               frame_gap=cfg.model.frame_gap,
                                               max_rewind_steps=cfg.model.max_rewind_steps,
                                               image_names=cfg.general.camera_names,
                                               annotation_list=cfg.model.sparse_annotation_list,
                                               task_name=cfg.general.task_name,
                                               video_eval=True)
        
        state_normalizer = get_normalizer_from_calculated(cfg.general.state_norm_path, self.device)

        # CLIP encoder
        clip_encoder = FrozenCLIPEncoder(cfg.encoders.vision_ckpt, self.device)
        vis_dim = 512
        txt_dim = 512
        
        subtask_model_path = Path(cfg.eval.ckpt_path) / cfg.eval.subtask_model
        stage_model_path = Path(cfg.eval.ckpt_path) / cfg.eval.stage_model
        
            
        if model_type == "sparse":
            num_classes = cfg.model.num_classes_sparse
        else:
            num_classes = cfg.model.num_classes_dense

        # --- reward_model ---
        subtask_model = SubtaskTransformer(d_model=cfg.model.d_model, 
                                  vis_emb_dim=vis_dim, 
                                  text_emb_dim=txt_dim,
                                  state_dim=cfg.model.state_dim,
                                  n_layers=cfg.model.n_layers,
                                  n_heads=cfg.model.n_heads,
                                  dropout=cfg.model.dropout,
                                  num_cameras=len(self.camera_names),
                                  ).to(self.device)
        stage_model = StageTransformer(d_model=cfg.model.d_model, 
                                  vis_emb_dim=vis_dim, 
                                  text_emb_dim=txt_dim,
                                  state_dim=cfg.model.state_dim,
                                  n_layers=cfg.model.n_layers,
                                  n_heads=cfg.model.n_heads,
                                  dropout=cfg.model.dropout,
                                  num_cameras=len(self.camera_names),
                                  num_classes_sparse=cfg.model.num_classes_sparse,
                                  num_classes_dense=cfg.model.num_classes_dense
                                  ).to(self.device)

        # Load checkpoints
        subtask_ckpt = torch.load(subtask_model_path, map_location=self.device)
        stage_ckpt = torch.load(stage_model_path, map_location=self.device)
        subtask_model.load_state_dict(subtask_ckpt["model"])
        stage_model.load_state_dict(stage_ckpt["model"])
        subtask_model.to(self.device)
        stage_model.to(self.device)
        subtask_model.eval(); stage_model.eval()

        # save path
        rollout_save_dir =  Path(self.save_dir) / "eval_video"
        rollout_save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_name = subtask_model_path.stem; ckpt_note_path = rollout_save_dir / "ckpt_note.txt"
        with open(ckpt_note_path, "w", encoding="utf-8") as f:
            f.write(f"subtask model: {ckpt_name}\n")
            
        OmegaConf.save(cfg, rollout_save_dir / "config.yaml")
        evaled_list = []

        for i in range(cfg.eval.run_times):
            ep_index = random.choice([idx for idx in val_eps if idx not in evaled_list])
            global_idx = val_eps.index(ep_index)
            evaled_list.append(ep_index)
            start_idx = dataset_val.episode_data_index["from"][global_idx].item()
            end_idx = dataset_val.episode_data_index["to"][global_idx].item() - 1
            gt_ep_result = []
            pred_ep_result = []
            pred_ep_smoothed = []
            pred_ep_conf = []
            x_offset = 0
            # x_offset = cfg.model.frame_gap * cfg.model.n_obs_steps
            eval_frame_gap = cfg.eval.eval_frame_gap
            smoother = RegressionConfidenceSmoother(value_range=(0.0, 1.0))
            print(f"[Eval Video] Evaluating episode_{ep_index}, progress: {i} / {cfg.eval.run_times}")

            # change to use tqdm
            for idx in tqdm(range(start_idx, end_idx, eval_frame_gap), desc=f"Processing episode {ep_index}"):
                data_point = dataset_val[idx]
                batch = adapt_lerobot_batch_sarm(data_point, camera_names=cfg.general.camera_names, eval_video=True)
                B, T = batch["image_frames"][self.camera_names[0]].shape[:2]
                img_list = []
                for key in self.camera_names:
                    imgs = batch["image_frames"][key].flatten(0, 1).to(self.device) # (B*T, C, H, W)
                    img_list.append(imgs)
                
                lang_strs = batch["tasks"]
                trg = batch["targets"].to(self.device)
                lens = batch["lengths"].to(self.device)
                state = batch["state"].to(self.device)
                state = state_normalizer.normalize(state)

                # CLIP encoding
                imgs_all = torch.cat(img_list, dim=0)  # (N * B * T, C, H, W)
                img_emb = clip_encoder.encode_image(imgs_all)  # (N * B * T, D)
                img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)  # (B, N, T, D)
                lang_emb = clip_encoder.encode_text(lang_strs) # lang_emb: (B, txt_dim)

                if cfg.model.no_state:
                    state = torch.zeros_like(state, device=self.device)
                
                stage_prob = stage_model(img_emb, lang_emb, state, lens, scheme=model_type).softmax(dim=-1)  # (B, T, num_classes)
                stage_idx = stage_prob.argmax(dim=-1)  # (B, T)
                stage_conf = stage_prob.gather(-1, stage_idx.unsqueeze(-1)).squeeze(-1)  # (B, T)
                
                # Inject stage prior to subtask model
                stage_onehot = F.one_hot(stage_idx, num_classes=stage_prob.size(-1)).float()  # (B, T, C)
                stage_emb = stage_onehot.unsqueeze(1)                          # (B, 1, T, C)
                subtask_pred = subtask_model(img_emb, lang_emb, state, lens, stage_emb)
                
                pred = torch.clip(subtask_pred + stage_idx.float(), 0, num_classes-1)  # (B, T)
                raw_item = pred[0, cfg.model.n_obs_steps].item()
        
                if model_type == "sparse":
                    raw_item_norm = normalize_sparse(raw_item)
                else:
                    raw_item_norm = normalize_dense(raw_item)
                
                conf_val = stage_conf[0, cfg.model.n_obs_steps].item()
                if idx >= (x_offset * eval_frame_gap):
                    smoothed_item = smoother.update(raw_item_norm, conf_val)
                else:
                    smoothed_item = raw_item_norm
                
                pred_ep_result.append(raw_item_norm)
                pred_ep_conf.append(conf_val)
                pred_ep_smoothed.append(smoothed_item)
                if dataset_type == "sparse":
                    gt_ep_result.append(normalize_sparse(trg[0, cfg.model.n_obs_steps].item()))
                else:
                    gt_ep_result.append(normalize_dense(trg[0, cfg.model.n_obs_steps].item()))

            # save results
            save_dir = plot_episode_result(ep_index, pred_ep_smoothed, gt_ep_result, x_offset, rollout_save_dir, frame_gap=eval_frame_gap, ep_conf=pred_ep_conf, ep_smoothed=pred_ep_smoothed)
            np.save(Path(save_dir) / "pred.npy", np.array(pred_ep_result))
            np.save(Path(save_dir) / "gt.npy", np.array(gt_ep_result))
            np.save(Path(save_dir) / "smoothed.npy", np.array(pred_ep_smoothed))
            print(f"[Eval Video] episode_{ep_index} making video...")
            chunk_id = ep_index // 1000
            root = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id # or change to your LEROBOT_LOCAL_DIR
            middle_video_dir = root / f"videos/chunk-{chunk_id:03d}/top_camera-images-rgb"
            try:
                produce_video(save_dir=rollout_save_dir, 
                              middle_video=middle_video_dir, 
                              episode_num=ep_index, 
                              x_offset=x_offset, 
                              frame_gap=eval_frame_gap)
            except Exception as e:
                print(f"[Eval Video] episode_{ep_index} video production failed: {e}")
            print(f"[Eval Video] episode_{ep_index} results saved to: {save_dir}, progress: {i+1} / {cfg.eval.run_times}")


    def eval_raw_data(self):
        import random
        cfg = self.cfg
        state_normalizer = get_normalizer_from_calculated(cfg.general.state_norm_path, self.device)

        # CLIP encoder
        clip_encoder = FrozenCLIPEncoder(cfg.encoders.vision_ckpt, self.device)
        vis_dim = 512
        txt_dim = 512

        subtask_model_path = Path(cfg.eval.ckpt_path) / cfg.eval.subtask_model
        stage_model_path = Path(cfg.eval.ckpt_path) / cfg.eval.stage_model
        
        
        model_type = cfg.eval.model_type
        if model_type == "sparse":
            num_classes = cfg.model.num_classes_sparse
        else:
            num_classes = cfg.model.num_classes_dense

        # --- reward_model ---
        subtask_model = SubtaskTransformer(d_model=cfg.model.d_model, 
                                  vis_emb_dim=vis_dim, 
                                  text_emb_dim=txt_dim,
                                  state_dim=cfg.model.state_dim,
                                  n_layers=cfg.model.n_layers,
                                  n_heads=cfg.model.n_heads,
                                  dropout=cfg.model.dropout,
                                  num_cameras=len(self.camera_names),
                                  ).to(self.device)
        stage_model = StageTransformer(d_model=cfg.model.d_model, 
                                  vis_emb_dim=vis_dim, 
                                  text_emb_dim=txt_dim,
                                  state_dim=cfg.model.state_dim,
                                  n_layers=cfg.model.n_layers,
                                  n_heads=cfg.model.n_heads,
                                  dropout=cfg.model.dropout,
                                  num_cameras=len(self.camera_names),
                                  num_classes_sparse=cfg.model.num_classes_sparse,
                                  num_classes_dense=cfg.model.num_classes_dense
                                  ).to(self.device)


        # Load checkpoints
        subtask_ckpt = torch.load(subtask_model_path, map_location=self.device)
        stage_ckpt = torch.load(stage_model_path, map_location=self.device)
        subtask_model.load_state_dict(subtask_ckpt["model"])
        stage_model.load_state_dict(stage_ckpt["model"])
        subtask_model.to(self.device)
        stage_model.to(self.device)
        subtask_model.eval(); stage_model.eval()

        # save path
        datetime_str = datetime.now().strftime("%Y.%m.%d/%H.%M.%S")
        rollout_save_dir = Path(self.save_dir) / "eval_video" / f"{datetime_str}"  # convert to Path first
        rollout_save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_name = subtask_model_path.stem; ckpt_note_path = rollout_save_dir / "ckpt_note.txt"
        with open(ckpt_note_path, "w", encoding="utf-8") as f:
            f.write(f"subtask model: {ckpt_name}\n")
            
        OmegaConf.save(cfg, rollout_save_dir / "config.yaml")
        
        # x_offset = cfg.model.frame_gap * cfg.model.n_obs_steps
        x_offset = 0
        data_dir = cfg.eval.raw_data_dir
        run_times = cfg.eval.raw_data_run_times
        # Get all valid episode paths
        all_episodes = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.startswith("episode_")
        ]
        eval_list = all_episodes
        
        random.seed(cfg.general.seed)
        # randomly select eval_list
        if len(all_episodes) >= run_times:
            eval_list = random.sample(all_episodes, run_times)
        else:
            raise ValueError(f"Not enough episodes in {data_dir} to sample {run_times} items.")

        for i in range(run_times):
            data_path = eval_list[i]
            pred_ep_result = []
            pred_ep_smoothed = []
            pred_ep_conf = []
            # randomly select 
            ep_index = os.path.basename(data_path)
            frame_num = get_frame_num(data_path)
            traj_joint_data = get_traj_data(data_path)
            eval_frame_gap = cfg.eval.eval_frame_gap
            smoother = RegressionConfidenceSmoother(value_range=(0.0, 1.0))
            print(f"[EVAL_RAW]: process {i+1}/{run_times} episode: {ep_index}")
            for idx in tqdm(range(0, frame_num, eval_frame_gap), desc=f"Processing data"):
                batch = get_frame_data_fast(path=data_path, 
                                    traj_joint_data=traj_joint_data, 
                                    idx=idx,
                                    n_obs_steps=cfg.model.n_obs_steps,
                                    frame_gap=cfg.model.frame_gap,
                                    max_rewind_steps=cfg.model.max_rewind_steps,
                                    camera_names=cfg.general.camera_names,
                                    device=self.device)
                
                B, T = batch["image_frames"][self.camera_names[0]].shape[:2]
                img_list = []
                for key in self.camera_names:
                    imgs = batch["image_frames"][key].flatten(0, 1).to(self.device) # (B*T, C, H, W)
                    img_list.append(imgs)
                
                lang_strs = ["fold the tshirt"]
                lens = torch.tensor([1+cfg.model.n_obs_steps], dtype=torch.int32, device=self.device)
                state = batch["state"].to(self.device)
                state = state_normalizer.normalize(state)
                
                # CLIP encoding
                imgs_all = torch.cat(img_list, dim=0)  # (N * B * T, C, H, W)
                img_emb = clip_encoder.encode_image(imgs_all)  # (N * B * T, D)
                img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)  # (B, N, T, D)
                lang_emb = clip_encoder.encode_text(lang_strs) # lang_emb: (B, txt_dim)

                if cfg.model.no_state:
                    state = torch.zeros_like(state, device=self.device)
                
                stage_prob = stage_model(img_emb, lang_emb, state, lens, scheme=cfg.eval.mode).softmax(dim=-1)  # (B, T, num_classes)
                stage_idx = stage_prob.argmax(dim=-1)  # (B, T)
                stage_conf = stage_prob.gather(-1, stage_idx.unsqueeze(-1)).squeeze(-1)  # (B, T)
                
                # Inject stage prior to subtask model
                stage_onehot = F.one_hot(stage_idx, num_classes=stage_prob.size(-1)).float()  # (B, T, C)
                stage_emb = stage_onehot.unsqueeze(1)                          # (B, 1, T, C)
                subtask_pred = subtask_model(img_emb, lang_emb, state, lens, stage_emb)

                pred = torch.clip(subtask_pred + stage_idx.float(), 0, num_classes-1)  # (B, T)
                raw_item = pred[0, cfg.model.n_obs_steps].item()
                if model_type == "sparse":
                    raw_item_norm = normalize_sparse(raw_item)
                else:
                    raw_item_norm = normalize_dense(raw_item)
                
                conf_val = stage_conf[0, cfg.model.n_obs_steps].item()
                if idx >= (x_offset * eval_frame_gap):
                    smoothed_item = smoother.update(raw_item_norm, conf_val)
                else:
                    smoothed_item = raw_item_norm
                
                pred_ep_result.append(raw_item_norm)
                pred_ep_conf.append(conf_val)
                pred_ep_smoothed.append(smoothed_item)

            # save results
            save_dir = plot_episode_result_raw_data(ep_index, pred_ep_result, x_offset, rollout_save_dir, frame_gap=eval_frame_gap, ep_conf=pred_ep_conf, ep_smoothed=pred_ep_smoothed)
            np.save(Path(save_dir) / "pred.npy", np.array(pred_ep_result))
            np.save(Path(save_dir) / "conf.npy", np.array(pred_ep_conf))
            np.save(Path(save_dir) / "smoothed.npy", np.array(pred_ep_smoothed))

            print(f"[Eval Video] episode_{ep_index} making video...")
            middle_video_dir = Path(f"{data_path}/top_camera-images-rgb.mp4")
            
            try:
                produce_video(save_dir=rollout_save_dir, 
                              middle_video=middle_video_dir, 
                              episode_num=ep_index, 
                              raw_data=True,
                              x_offset=x_offset, 
                              frame_gap=eval_frame_gap)
            except Exception as e:
                print(f"[Eval Video] episode_{ep_index} video production failed: {e}")
            
            print(f"[Eval Video] episode_{ep_index} results saved to: {save_dir}")


    