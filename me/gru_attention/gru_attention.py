import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import copy
import bisect
from typing import Optional, Union, Text
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from gru_attention.gru_attention_dataset import GRUAttentionDatasetH

# Import Qlib related modules
from qlib.log import get_module_logger
from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import get_or_create_path


class MarketContextAttentionModel(nn.Module):
    """
    A self-attention layer to extract a global market context vector from all stocks at each time step.
    """
    def __init__(self, input_size, context_size):
        super().__init__()
        # Linear layers to generate Query, Key, Value
        # Here, a unified input_size is used as input and mapped to context_size
        self.query_layer = nn.Linear(input_size, context_size)
        self.key_layer = nn.Linear(input_size, context_size)
        self.value_layer = nn.Linear(input_size, context_size)

    def forward(self, stock_features, stock_mask: Optional[torch.Tensor] = None):
        """
        Input shape: (batch_size, stocks, features)
        stock_mask shape: (batch_size, stocks) with 1 for valid stocks, 0 for padded/absent
        Output shape: (batch_size, context_size) - market context vector
        """
        # 1. Generate Query, Key, Value
        Q = self.query_layer(stock_features) # (batch_size, stocks, context_size)
        K = self.key_layer(stock_features)   # (batch_size, stocks, context_size)
        V = self.value_layer(stock_features) # (batch_size, stocks, context_size)

        # 2. Calculate attention scores (Scaled Dot-Product Attention)
        # scores shape: (batch_size, stocks, stocks)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (K.size(-1)**0.5)
        if stock_mask is not None:
            # Mask keys (bool): set scores to very negative where key is invalid
            # key_mask shape -> (batch, 1, stocks)
            key_mask = stock_mask.unsqueeze(1)
            scores = scores.masked_fill(~key_mask, -1e9)
        
        # 3. Normalize scores to get attention weights
        attention_weights = F.softmax(scores, dim=-1) # (batch_size, stocks, stocks)

        # 4. Apply attention weights to Value to get context-aware representation for each stock
        # context_per_stock shape: (batch_size, stocks, context_size)
        context_per_stock = torch.bmm(attention_weights, V)

        # 5. Aggregate context representations of all stocks to form a single market context vector
        # Use masked mean across the query dimension so absent stocks do not contribute
        if stock_mask is not None:
            query_mask = stock_mask.unsqueeze(-1)  # (batch, stocks, 1) bool
            masked_ctx = context_per_stock.masked_fill(~query_mask, 0.0)
            masked_sum = masked_ctx.sum(dim=1)
            denom = query_mask.sum(dim=1).clamp_min(1)  # count of valid stocks (int)
            market_context_vector = masked_sum / denom.to(masked_sum.dtype).clamp_min(1e-6)
        else:
            market_context_vector = context_per_stock.mean(dim=1) # (batch_size, context_size)

        return market_context_vector
    
class GRUWithAttentionModel(nn.Module):
    def __init__(self, input_size: int, output_size: int, context_size: int = 20, gru_hidden_size: int = 64, gru_num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        
        self.input_size = input_size
        self.gru_hidden_size = gru_hidden_size
        self.context_size = context_size # Dimension of market context vector
        
        # 1. Market Context Attention Layer: Capture cross-sectional information at each time step
        self.market_context_attention = MarketContextAttentionModel(input_size, context_size)
        
        # 2. LayerNorm: Normalize after concatenating original features and context, to stabilize GRU input
        # Input feature dimension for GRU is (original_feature_dim + context_size)
        self.layer_norm = nn.LayerNorm(input_size + context_size)
        
        # 3. GRU Layer: Each GRU unit now receives (original_features + market_context)
        gru_dropout = dropout if gru_num_layers > 1 else 0
        self.gru = nn.GRU(
            input_size=input_size + context_size, # GRU input dimension is concatenation of original features and context
            hidden_size=gru_hidden_size, 
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=gru_dropout
        )
        
        # 4. Fully Connected Output Layer: Map GRU's last time step output for each stock to prediction
        self.output_head = nn.Sequential(
            nn.Linear(gru_hidden_size, gru_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_size // 2, output_size)
        )
        
        # 5. Call weight initialization method
        self._init_weights()

    def _init_weights(self):
        """
        Initialize model weights with appropriate methods.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Xavier uniform distribution to initialize linear layer weights
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # Initialize bias terms to zero
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                # Initialize GRU weights
                # Use Xavier uniform distribution for input-to-hidden weights
                for name, param in m.named_parameters():
                    if 'weight_ih' in name: # Input-to-hidden weights
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name: # Hidden-to-hidden weights
                        nn.init.orthogonal_(param)
                    elif 'bias' in name: # Bias terms
                        nn.init.constant_(param, 0)

    def log_gradients(self):
        """
        日志记录：仅在梯度异常（消失/爆炸）时输出每层的 L2 norm / min / max。
        """
        abnormal = []
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                norm = grad.norm(2)  # L2 norm

                # # ---- 新增的特定层日志（用于诊断）----
                # norm_val = norm.item()
                # min_val = grad.min().item()
                # max_val = grad.max().item()
                # if name == "cross_feature_attn.key.weight" or name == "cross_feature_attn.query.weight":
                #     self.logger.info(f"[INFO_GRAD] {name} | L2 norm={norm_val:.2e} | min={min_val:.2e} | max={max_val:.2e}")
                # # ---- 结束新增 ----

                # 判断阈值
                if norm < 1e-6 and "bias" not in name:
                    norm_val = norm.item()
                    min_val = grad.min().item()
                    max_val = grad.max().item()
                    # self.logger.info(f"[vanish] {name} | L2 norm={norm_val:.2e} | min={min_val:.2e} | max={max_val:.2e}")
                    abnormal.append(("vanish", name, norm_val, min_val, max_val))
                elif norm > 1e2:
                    norm_val = norm.item()
                    min_val = grad.min().item()
                    max_val = grad.max().item()
                    # self.logger.info(f"[explode] {name} | L2 norm={norm_val:.2e} | min={min_val:.2e} | max={max_val:.2e}")
                    abnormal.append(("explode", name, norm_val, min_val, max_val))
        return abnormal

    def forward(self, x, feature_mask: Optional[torch.Tensor] = None):
        """
        Forward pass of the model.
        Input shape: (batch_size, days, stocks, features)
        feature_mask shape: (batch_size, days, stocks), 1 for valid entries
        Output shape: (batch_size, stocks, output_size) - prediction for each stock
        """
        batch_size, days, stocks, features = x.shape
        
        # Assert input shape correctness
        assert features == self.input_size, f"Input feature dimension ({features}) does not match model definition ({self.input_size})"
        
        # List to store augmented features for each time step, for each stock
        # Each element in the list will have shape: (batch_size, stocks, input_size + context_size)
        augmented_sequence_per_time = []

        for t in range(days):
            # 1. Extract features for all stocks at current time step: (batch_size, stocks, features)
            stock_data_at_t = x[:, t, :, :]
            mask_t = None
            if feature_mask is not None:
                mask_t = feature_mask[:, t, :]
            
            # 2. Capture market context information: (batch_size, context_size)
            market_context_t = self.market_context_attention(stock_data_at_t, stock_mask=mask_t)
            
            # 3. Replicate market context for each stock and concatenate with original features
            # replicated_context_t shape: (batch_size, stocks, context_size)
            replicated_context_t = market_context_t.unsqueeze(1).expand(-1, stocks, -1)
            
            # augmented_stock_features_t shape: (batch_size, stocks, features + context_size)
            augmented_stock_features_t = torch.cat((stock_data_at_t, replicated_context_t), dim=-1)
            
            # 4. Apply LayerNorm to the concatenated features
            augmented_stock_features_t = self.layer_norm(augmented_stock_features_t)
            
            augmented_sequence_per_time.append(augmented_stock_features_t)
        
        # 5. Stack augmented features for all time steps, preparing for GRU
        # stacked_augmented_sequence shape: (batch_size, days, stocks, input_size + context_size)
        stacked_augmented_sequence = torch.stack(augmented_sequence_per_time, dim=1)

        # 6. Reshape GRU input: Permute and then merge batch_size and stocks dimensions
        # stacked_augmented_sequence shape: (batch_size, stocks, days, input_size + context_size)
        stacked_augmented_sequence = stacked_augmented_sequence.permute(0, 2, 1, 3)
        # gru_input shape: (batch_size * stocks, days, input_size + context_size)
        gru_input = stacked_augmented_sequence.reshape(-1, days, features + self.context_size)
        
        # 7. Input to GRU network
        # gru_output shape: (batch_size * stocks, days, gru_hidden_size)
        gru_output, _ = self.gru(gru_input)

        # 8. Extract GRU's last time step output (for each stock)
        # last_step_output_flat shape: (batch_size * stocks, gru_hidden_size)
        last_step_output_flat = gru_output[:, -1, :]

        # 9. Reshape back to (batch_size, stocks, gru_hidden_size) for per-stock prediction
        last_step_output_per_stock = last_step_output_flat.reshape(batch_size, stocks, self.gru_hidden_size)

        # 10. Pass through output head to get classification results for each stock
        # logits shape: (batch_size, stocks, output_size)
        logits = self.output_head(last_step_output_per_stock)

        return logits


class GRUAttention(Model):
    """
    Qlib Custom GRU + Attention Model Class
    This class inherits from qlib.model.base.Model and implements training and prediction logic.
    """
    def __init__(
        self,
        context_size: int = 20,
        gru_hidden_size: int = 64,
        gru_num_layers: int = 1,
        lr: float = 0.001,
        epochs: int = 200,
        batch_size: int = 2048,
        dropout: float = 0.2,
        early_stop: int = 10,
        regression: bool = True,
        seed: Optional[int] = None,
        step_len: int = 20,
        min_valid_days_in_window: Optional[int] = None,
        min_tail_consecutive_days: Optional[int] = None,
        use_amp: bool = True,
        **kwargs,
    ):
        self.logger = get_module_logger("GRUAttention")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.context_size = context_size
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.early_stop = early_stop
        self.regression = regression
        self.seed = seed
        self.step_len = step_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Optional k/c constraints applied to label/pred gating
        # k: minimum valid days in the lookback window to be eligible
        # c: minimum consecutive valid days at the tail (including T) to be eligible
        self.min_valid_days_in_window = min_valid_days_in_window
        self.min_tail_consecutive_days = min_tail_consecutive_days
        self.use_amp = use_amp
        self.scaler = GradScaler(device='cuda', enabled=self.use_amp)
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else None

        if regression:
            self.output_size = 1
            self.criterion = nn.MSELoss()
        else:
            self.output_size = 2
            self.criterion = nn.CrossEntropyLoss()

        self.logger.info(
            "GRUAttention parameters setting:"
            "\noutput_size : {}"
            "\ncontext_size : {}"
            "\ngru_hidden_size : {}"
            "\ngru_num_layers : {}"
            "\ndropout : {}"
            "\nepochs : {}"
            "\nlr : {}"
            "\nbatch_size : {}"
            "\nearly_stop : {}"
            "\ndevice : {}"
            "\nregression : {}"
            "\nseed : {}"
            "\nstep_len : {}"
            "\nuse_amp : {}"
            "\ndtype : {}"
            "\nmin_valid_days_in_window (k) : {}"
            "\nmin_tail_consecutive_days (c) : {}".format(
                self.output_size,
                context_size,
                gru_hidden_size,
                gru_num_layers,
                dropout,
                epochs,
                lr,
                batch_size,
                early_stop,
                self.device,
                regression,
                seed,
                step_len,
                self.use_amp,
                self.dtype,
                self.min_valid_days_in_window,
                self.min_tail_consecutive_days,
            )
        )
        self.fitted = False

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def _calculate_loss(self, outputs, targets, label_mask: Optional[torch.Tensor] = None):
        """Calculate loss based on regression or classification mode"""
        if self.regression:
            # For regression: outputs (batch_size, stocks, 1), targets (batch_size, stocks)
            outputs_flat = outputs.view(-1)  # Flatten to 1D
            targets_flat = targets.view(-1)   # Flatten to 1D

            # Filter out invalid targets using provided label_mask (preferred) or NaN check
            if label_mask is not None:
                mask = label_mask.reshape(-1)
            else:
                mask = ~torch.isnan(targets_flat) & ~torch.isnan(outputs_flat)
            if mask.sum() == 0:
                raise ValueError("All target or output values are NaN")
            
            return self.criterion(outputs_flat[mask], targets_flat[mask])
        else:
            # For classification: outputs (batch_size, stocks, num_classes), targets (batch_size, stocks)
            outputs_flat = outputs.view(-1, self.output_size)  # (batch_size * stocks, num_classes)
            targets_flat = targets.view(-1).long()  # (batch_size * stocks)
            # Filter out invalid targets
            if label_mask is not None:
                mask = label_mask.reshape(-1)
            else:
                # Assume all given labels are valid in classification if no mask provided
                mask = torch.ones_like(targets_flat, dtype=torch.bool)
            if mask.sum() == 0:
                raise ValueError("All target or output values are NaN")
            
            return self.criterion(outputs_flat[mask], targets_flat[mask])
        
    def _get_combined_label_mask(self, feature_mask_batch, label_mask_batch):
        ''' Apply optional k/c constraints to label mask (in addition to dataset-provided mask) '''
        combined_label_mask = label_mask_batch
        if feature_mask_batch is not None:
            # last day presence mask (batch, stocks)
            if combined_label_mask is None:
                combined_label_mask = feature_mask_batch[:, -1, :].clone()
            # k: minimum valid days in window
            if self.min_valid_days_in_window is not None:
                valid_days = feature_mask_batch.sum(dim=1)  # (batch, stocks), int
                cond_k = valid_days >= int(self.min_valid_days_in_window)
                combined_label_mask = combined_label_mask & cond_k
            # c: minimum tail consecutive valid days
            if self.min_tail_consecutive_days is not None and int(self.min_tail_consecutive_days) > 1:
                fm_rev = feature_mask_batch.flip(dims=[1]).to(dtype=torch.float32)
                tail_run = torch.cumprod(fm_rev, dim=1).sum(dim=1)  # (batch, stocks), float
                cond_c = tail_run >= float(int(self.min_tail_consecutive_days))
                combined_label_mask = combined_label_mask & cond_c
        return combined_label_mask
        
    def fit(self, dataset: DatasetH, evals_result=None, save_path=None):
        if not isinstance(dataset, GRUAttentionDatasetH):
            raise ValueError("Dataset must be an instance of GRUAttentionDatasetH for GRU with Attention model.")

        # Prepare training and validation data using our custom dataset
        train_ds = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        valid_ds = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=self.batch_size, shuffle=False)

        if len(train_ds) == 0:
            raise ValueError("Training data is empty. Please check your dataset preparation.")

        input_size = train_ds[0][0].shape[-1]
        self.model = GRUWithAttentionModel(
            input_size=input_size,
            output_size=self.output_size,
            context_size=self.context_size,
            gru_hidden_size=self.gru_hidden_size,
            gru_num_layers=self.gru_num_layers,
            dropout=self.dropout,
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        best_val_loss = float("inf")
        best_epoch = 0
        best_param = copy.deepcopy(self.model.state_dict())

        if evals_result is None:
            evals_result = {"train": [], "valid": []}
        else:
            evals_result["train"] = []
            evals_result["valid"] = []

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            total_loss = 0
            num_batches = 0

            for batch in train_loader:
                # Unpack batch supporting masks: (X, y, feature_mask, label_mask) or (X, y)
                if len(batch) == 4:
                    X_batch, y_batch, feature_mask_batch, label_mask_batch = batch
                    feature_mask_batch = feature_mask_batch.to(self.device)
                    label_mask_batch = label_mask_batch.to(self.device)
                else:
                    X_batch, y_batch = batch
                    feature_mask_batch = None
                    label_mask_batch = None
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                with autocast(device_type='cuda', dtype=self.dtype, enabled=self.use_amp):
                    outputs = self.model(X_batch, feature_mask=feature_mask_batch)  # (batch, stocks, output)
                    # Apply optional k/c constraints to label mask (in addition to dataset-provided mask)
                    combined_label_mask = self._get_combined_label_mask(feature_mask_batch, label_mask_batch)
                    # Calculate loss
                    loss = self._calculate_loss(outputs, y_batch, label_mask=combined_label_mask)
                self.scaler.scale(loss).backward()
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # 监控梯度
                abnormal = self.model.log_gradients()
                if abnormal:  # 返回列表形式 [("vanish", name, norm, min, max), ...]
                    vanish_counter = sum(1 for item in abnormal if item[0] == "vanish")
                    explode_counter = sum(1 for item in abnormal if item[0] == "explode")
                if vanish_counter > 5:
                    self.logger.warning(f"🚨 梯度消失{vanish_counter} 次")
                if explode_counter > 5:
                    self.logger.warning(f"🚨 梯度爆炸{explode_counter}次")

                total_loss += loss.item()
                num_batches += 1

            avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
            evals_result["train"].append(-avg_train_loss)

            # Validation phase
            self.model.eval()
            valid_loss = 0
            valid_batches = 0

            with torch.no_grad():
                for batch in valid_loader:
                    if len(batch) == 4:
                        X_batch, y_batch, feature_mask_batch, label_mask_batch = batch
                        feature_mask_batch = feature_mask_batch.to(self.device)
                        label_mask_batch = label_mask_batch.to(self.device)
                    else:
                        X_batch, y_batch = batch
                        feature_mask_batch = None
                        label_mask_batch = None
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    with autocast(device_type='cuda', dtype=self.dtype, enabled=self.use_amp):
                        outputs = self.model(X_batch, feature_mask=feature_mask_batch)
                        # Apply optional k/c constraints during validation
                        combined_label_mask = self._get_combined_label_mask(feature_mask_batch, label_mask_batch)
                        loss = self._calculate_loss(outputs, y_batch, label_mask=combined_label_mask)

                    valid_loss += loss.item()
                    valid_batches += 1

            avg_valid_loss = valid_loss / valid_batches if valid_batches > 0 else float("inf")
            evals_result["valid"].append(-avg_valid_loss)

            if avg_valid_loss < best_val_loss:
                best_val_loss = avg_valid_loss
                stop_steps = 0
                best_epoch = epoch + 1
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

            self.logger.info(
                f"Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}"
            )

        self.logger.info("best val loss: %.6lf @ %d" % (best_val_loss, best_epoch))
        self.model.load_state_dict(best_param)

        if save_path:
            torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

        self.fitted = True

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test") -> pd.Series:
        if not self.fitted:
            raise ValueError("model is not fitted yet! Please call fit() first.")

        if not isinstance(dataset, GRUAttentionDatasetH):
            raise ValueError("Dataset must be an instance of GRUAttentionDatasetH for GRU with Attention model.")

        # Prepare test data
        test_ds = dataset.prepare(segment, col_set=["feature"], data_key=DataHandlerLP.DK_I)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        if len(test_ds) == 0:
            raise ValueError(f"Test data for segment {segment} is empty. Please check your dataset preparation.")

        preds = []
        present_masks = []  # last-day presence masks to align length with infer_index
        combined_masks = []  # optional k/c gating masks to NaN-out values after alignment
        self.model.eval()

        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 2:
                    X_batch, feature_mask_batch = batch
                    feature_mask_batch = feature_mask_batch.to(self.device)
                else:
                    X_batch = batch[0]
                    feature_mask_batch = None
                X_batch = X_batch.to(self.device)
                pred = (
                    self.model(X_batch, feature_mask=feature_mask_batch).detach().cpu().numpy()
                )  # (batch_size, stocks, output_size)
                # Collect masks
                if feature_mask_batch is not None:
                    # Presence on the last day (T) — used to align with infer_index
                    last_day_presence = feature_mask_batch[:, -1, :]
                    present_masks.append(last_day_presence.detach().cpu().numpy())
                    # Optional k/c gating (subset of presence) — used to set NaN on values but not to slice index
                    combined_label_mask = self._get_combined_label_mask(feature_mask_batch, None)
                    combined_masks.append(combined_label_mask.detach().cpu().numpy())
                preds.append(pred)

        # Concatenate all predictions and masks
        all_preds = np.concatenate(preds, axis=0)  # (num_samples, stocks, output_size)

        # Determine presence mask for alignment (defaults to all True if no mask provided)
        if present_masks:
            presence = np.concatenate(present_masks, axis=0)  # (num_samples, stocks)
        else:
            presence = np.ones(all_preds.shape[:2], dtype=bool)

        presence_flat = presence.reshape(-1)
        infer_index = dataset.infer_index  # Do NOT slice infer_index with k/c; it's already built from presence

        # Flatten predictions by presence to match infer_index length
        if self.output_size == 1:
            values = all_preds.reshape(-1)[presence_flat]
            # Apply optional k/c gating as NaN without changing index length
            if combined_masks:
                combined = np.concatenate(combined_masks, axis=0).reshape(-1)
                combined_on_present = combined[presence_flat]
                values[~combined_on_present] = np.nan
            return pd.Series(values, index=infer_index)
        else:
            values = all_preds.reshape(-1, self.output_size)[presence_flat]
            # Apply optional k/c gating as NaN without changing index length
            if combined_masks:
                combined = np.concatenate(combined_masks, axis=0).reshape(-1)
                combined_on_present = combined[presence_flat]
                values[~combined_on_present, :] = np.nan
            columns = [f"score_class_{i}" for i in range(self.output_size)]
            return pd.DataFrame(values, index=infer_index, columns=columns)

