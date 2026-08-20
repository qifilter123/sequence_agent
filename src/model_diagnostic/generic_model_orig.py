import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualGRUBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1, use_residual=True):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.use_residual = use_residual

        # 安全防线：防止外部调用错误导致张量维度 Crash
        if self.use_residual and input_dim != hidden_dim:
            raise ValueError(
                f"Cannot use residual connection when input_dim ({input_dim}) != hidden_dim ({hidden_dim})"
            )

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.drop(self.ln(out))
        return out + x if self.use_residual else out


class ContinuousTimeEmbedding(nn.Module):
    """Learnable Sinusoidal Time Embedding for continuous duration (dt)."""

    def __init__(self, out_dim: int = 16):
        super().__init__()
        self.out_dim = out_dim
        assert out_dim % 2 == 0, "Time embedding dimension must be even."
        # Initialize learnable frequencies (omega) and phases (phi)
        self.omega = nn.Parameter(torch.randn(out_dim // 2))
        self.phi = nn.Parameter(torch.randn(out_dim // 2))

    def forward(self, dt):
        # dt shape: [B, L] -> scaled shape: [B, L, out_dim // 2]
        scaled = dt.unsqueeze(-1) * self.omega + self.phi
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)


class ForwardSeqEncoder(nn.Module):
    """Forward-only GRU + Causal last-hidden state embedding with Continuous Time2Vec Embedding."""

    def __init__(self, cfg):
        super().__init__()
        self.hidden_dim = cfg.hidden_dim

        self.base_dim = cfg.input_base_dim
        self.time_emb_dim = cfg.time_emb_dim
        self.combo_dim = cfg.combo_dim
        self.sw_classes = cfg.sw_classes
        self.is_new_classes = cfg.is_new_classes

        self.time_emb = ContinuousTimeEmbedding(out_dim=self.time_emb_dim)

        self.feature_corr = nn.Sequential(
            nn.Linear(self.base_dim + self.time_emb_dim, 32),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(32, self.combo_dim),
            nn.ReLU()
        )

        # 3. 动态推断 GRU 输入维度 (绕过缺乏 LazyGRU 的限制)
        gru_input_dim = self.base_dim + self.time_emb_dim + self.combo_dim

        self.fwd_stack = nn.ModuleList([
            ResidualGRUBlock(
                input_dim=gru_input_dim if i == 0 else self.hidden_dim,
                hidden_dim=self.hidden_dim,
                dropout=cfg.dropout,
                use_residual=(i > 0)
            )
            for i in range(cfg.num_layers)
        ])

        # 预测头定义
        self.v_head = nn.Linear(self.hidden_dim, 1)
        self.sw_head = nn.Linear(self.hidden_dim, self.sw_classes)
        self.amt_head = nn.Linear(self.hidden_dim, 1)
        self.is_new_head = nn.Linear(self.hidden_dim, self.is_new_classes)

    def _enrich(self, x_full):
        # 语法糖优化：使用 [..., 0] 提取最后一个维度的第 0 位
        dt = x_full[..., 0]
        t_emb = self.time_emb(dt)

        x_with_time = torch.cat([x_full, t_emb], dim=-1)
        sw_combo = self.feature_corr(x_with_time)

        return torch.cat([x_with_time, sw_combo], dim=-1)

    '''def _enrich(self, x_full):
        dt = x_full[..., 0]
        dt_log = torch.log1p(dt)

        # Only TimeEmbedding sees log-scaled dt.
        t_emb = self.time_emb(dt_log)

        # Direct numeric path keeps raw dt.
        x_with_time = torch.cat([x_full, t_emb], dim=-1)

        sw_combo = self.feature_corr(x_with_time)

        return torch.cat([x_with_time, sw_combo], dim=-1)'''

    def _run_fwd(self, x):
        h = x
        for block in self.fwd_stack:
            h = block(h)
        return h

    def encode(self, x_full, mask=None):
        return self._run_fwd(self._enrich(x_full))

    def predict_nextstep(self, fwd_out):
        h_head = fwd_out[:, :-1, :]
        return {
            "v_pred": self.v_head(h_head).squeeze(-1),
            "sw_logits": self.sw_head(h_head),
            "amt_pred": self.amt_head(h_head).squeeze(-1),
            "is_new_logits": self.is_new_head(h_head)
        }

    def extract_embedding(self, x_full, mask):
        fwd_out = self.encode(x_full, mask)
        valid_len = torch.clamp(mask.sum(dim=1).long(), min=1)

        # 4. 性能与结构优化：使用高级索引(Advanced Indexing)取代复杂的 expand().gather()
        batch_indices = torch.arange(fwd_out.size(0), device=fwd_out.device)
        last = fwd_out[batch_indices, valid_len - 1]

        return F.normalize(last, p=2, dim=-1), fwd_out
