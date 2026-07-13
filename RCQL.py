import os
import warnings

# 1. Suppress all Python warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# 2. Suppress hardware-specific and logging-heavy backends
os.environ["MKLDNN_VERBOSE"] = "0"
os.environ["MKL_VERBOSE"] = "0"
os.environ["NNPACK_VERBOSE"] = "0"

# 3. Import torch and immediately configure backends
import torch

# try:
#     torch.backends.nnpack.enabled = False
#     torch.backends.cudnn.enabled = False
# except Exception:
#     pass

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import collections
import random
from Agent import Agent
from ValueBC import temperature_scaled_bc_loss


class NatureCNNEncoder(nn.Module):
    def __init__(self, in_channels=3, img_size=64):
        super(NatureCNNEncoder, self).__init__()
        # Standard Nature CNN convolutional stack
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate flatten size dynamically (1024 for 64x64 inputs)
        dummy_input = torch.zeros(1, in_channels, img_size, img_size)
        with torch.no_grad():
            dummy_out = self.conv3(self.conv2(self.conv1(dummy_input)))
            flatten_size = dummy_out.numel()

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(flatten_size, 512)

    def forward(self, x):
        # Swapped from ReLU to ELU to prevent dead units during BPTT
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = self.flatten(x)
        x = F.elu(self.fc(x))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels, act=F.relu):
        super().__init__()
        self.act = act
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.act(x)
        out = self.conv1(out)
        out = self.act(out)
        out = self.conv2(out)
        return x + out


class ConvLayerNorm(nn.Module):
    """LayerNorm over the channel dim of a (B, C, H, W) feature map
    (channels-last style, per spatial position)."""

    def __init__(self, channels):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act=F.relu):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels, act=act)
        self.res2 = ResidualBlock(out_channels, act=act)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaCNN(nn.Module):
    def __init__(self, in_channels=3, img_size=64, depths=[16, 32, 32], out_size=256,
                 act=F.relu, pool="flatten", layer_norm=False):
        super().__init__()
        # act=F.elu gives the "impala_elu" ablation variant: the Nature
        # encoder was deliberately moved to ELU to avoid dead units during
        # BPTT, and default-impala reintroduced ReLU everywhere.
        # pool="avg" + layer_norm=True gives the "impoola" variant: global
        # average pooling replaces the flatten (Impoola architecture) and a
        # channelwise LayerNorm follows each conv block.
        blocks = []
        ch = in_channels
        for out_channels in depths:
            blocks.append(ImpalaBlock(ch, out_channels, act=act))
            if layer_norm:
                blocks.append(ConvLayerNorm(out_channels))
            ch = out_channels
        self.blocks = nn.Sequential(*blocks)

        act_mod = nn.ELU() if act is F.elu else nn.ReLU()
        if pool == "avg":
            # Global average pooling: fc input is just the final channel
            # count (32 by default), independent of spatial resolution.
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), act_mod,
                nn.Linear(depths[-1], out_size), act_mod
            )
        else:
            # For a 64x64 input, 3 max pools of stride 2 yield an 8x8 spatial
            # resolution. 8 * 8 * 32 (final depth) = 2048. Computed
            # dynamically for other sizes.
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            with torch.no_grad():
                flatten_size = self.blocks(dummy).numel()
            self.fc = nn.Sequential(
                nn.Flatten(), act_mod, nn.Linear(flatten_size, out_size), act_mod
            )

    def forward(self, x):
        # Ensure input is scaled to [0, 1] if passing raw pixels
        x = self.blocks(x)
        x = self.fc(x)
        return x


class RecurrentQNetwork(nn.Module):
    def __init__(self, action_dim, in_channels=3, img_size=64, hidden_dim=512,
                 encoder="impala", encoder_width=1, encoder_chunks=1):
        super(RecurrentQNetwork, self).__init__()
        # encoder_chunks > 1: activation-checkpoint the CNN forward in that
        # many frame chunks — conv activations are recomputed chunk-by-chunk
        # during backward instead of stored, so the live graph shrinks by
        # roughly that factor (needed for width-2 encoders on a 16GB card)
        # at the cost of one extra encoder forward per update. Exact same
        # gradients; no_grad paths (burn-in, target net, acting) unaffected.
        self.encoder_chunks = encoder_chunks
        # IMPALA ResNet encoder by default; "nature" keeps the old Nature CNN
        # (needed to load pre-impala checkpoints). Both emit 512-d features —
        # buffer replay code (refresh_hidden_states / compute_td_errors)
        # depends on that width.
        # encoder_width = Impoola-paper tau: channel depths [16,32,32]*tau
        # (their recommended config is tau=2). Ignored by the nature encoder.
        depths = [16 * encoder_width, 32 * encoder_width, 32 * encoder_width]
        if encoder == "impala":
            self.encoder = ImpalaCNN(in_channels, img_size, depths=depths,
                                     out_size=512)
        elif encoder == "impala_elu":
            self.encoder = ImpalaCNN(in_channels, img_size, depths=depths,
                                     out_size=512, act=F.elu)
        elif encoder == "impoola":
            self.encoder = ImpalaCNN(in_channels, img_size, depths=depths,
                                     out_size=512, pool="avg", layer_norm=True)
        elif encoder == "impoola_elu":
            self.encoder = ImpalaCNN(in_channels, img_size, depths=depths,
                                     out_size=512, act=F.elu, pool="avg",
                                     layer_norm=True)
        elif encoder == "nature":
            self.encoder = NatureCNNEncoder(in_channels, img_size)
        else:
            raise ValueError(f"Unknown encoder: {encoder}")
        self.lstm = nn.LSTM(512, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim + 1)

    def forward(self, x, hidden=None, features=None):
        batch_size, seq_len, c, h, w = x.size()

        if features is None:
            # Flatten batch and sequence for the CNN forward pass
            x_flat = x.reshape(batch_size * seq_len, c, h, w)
            if self.encoder_chunks > 1 and torch.is_grad_enabled() and self.training:
                features = torch.cat([
                    torch.utils.checkpoint.checkpoint(
                        self.encoder, chunk, use_reentrant=False
                    )
                    for chunk in x_flat.chunk(self.encoder_chunks)
                ])
            else:
                features = self.encoder(x_flat)
            features = features.reshape(batch_size, seq_len, -1)

        lstm_out, hidden = self.lstm(features, hidden)
        out = self.fc(lstm_out)

        # Dueling Q-Network stream extraction
        adv = out[:, :, :-1]
        adv = adv - adv.mean(dim=-1, keepdim=True)
        v = out[:, :, -1:]
        q = v + adv

        return q, v, adv, hidden


class RCQLAgent(Agent):
    def __init__(
        self,
        obs_dim,
        action_dim,
        name="RCQL",
        save_dir="results",
        device_name="cpu",
        hidden_dim=512,
        lr=3e-4,
        epsilon=0.1,
        encoder="impala",
        encoder_width=1,
        encoder_chunks=1,
        amp=False,
    ):
        super().__init__(obs_dim, action_dim, name, save_dir, device_name)

        in_channels = obs_dim[0]
        img_size = obs_dim[1]

        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        # bf16 autocast for the update forwards (--amp). bf16 needs no
        # GradScaler (fp32 exponent range); params/grads/optimizer stay fp32.
        self.amp = amp

        self.q_net = RecurrentQNetwork(
            action_dim, in_channels, img_size, hidden_dim, encoder=encoder,
            encoder_width=encoder_width, encoder_chunks=encoder_chunks,
        ).to(self.device_name)
        self.q_target = copy.deepcopy(self.q_net)
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.gamma = 0.99
        self.tau = 0.005
        self.cql_alpha = 1.0
        # Weight on the demo imitation (BC) loss; annealable from the runner
        # like cql_alpha. Scales the gradient only — reported loss stays raw.
        self.bc_weight = 1.0

        self.q_hidden = None
        self.last_q = None
        self.replay_buffer = collections.deque(maxlen=1000)

    def _autocast(self):
        """bf16 autocast context for update-path forwards; no-op unless
        --amp and CUDA. Backwards/steps always run OUTSIDE this context —
        autograd replays the recorded (bf16) ops and accumulates fp32 grads."""
        enabled = self.amp and str(self.device_name).startswith("cuda")
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled)

    def reset_hidden(self):
        self.q_hidden = None

    def get_hidden_snapshot(self, idx=0):
        """Returns the current actor LSTM state for batch slot idx as a
        (2, hidden_dim) fp16 tensor (state BEFORE the next observation is
        processed), or None at episode start."""
        if self.q_hidden is None:
            return None
        h, c = self.q_hidden
        return torch.stack([h[0, idx], c[0, idx]]).detach().to(torch.float16)

    def reset_hidden_index(self, idx):
        """Zeroes the recurrent state of one batch slot (one parallel env)."""
        if self.q_hidden is not None:
            h, c = self.q_hidden
            h[:, idx, :] = 0.0
            c[:, idx, :] = 0.0

    def _make_hidden(self, init_hidden):
        """Converts a (B, 2, hidden_dim) stored-state tensor into an LSTM (h0, c0) tuple."""
        if init_hidden is None:
            return None
        h = init_hidden[:, 0, :].unsqueeze(0).contiguous().float()
        c = init_hidden[:, 1, :].unsqueeze(0).contiguous().float()
        return (h, c)

    def _normalize(self, obs):
        if obs.dtype == torch.uint8 or obs.max() > 1.0:
            return obs.float() / 255.0
        return obs.float()

    def _ensure_channel_first(self, x):
        if x.shape[-1] in [1, 3] and x.shape[-3] not in [1, 3]:
            ndim = x.ndim
            if ndim == 3:
                return x.permute(2, 0, 1)
            if ndim == 4:
                return x.permute(0, 3, 1, 2)
            if ndim == 5:
                return x.permute(0, 1, 4, 2, 3)
        return x

    def act(self, observations: torch.Tensor, deterministic: bool = False):
        self.q_net.eval()
        with torch.no_grad():
            if observations.ndim == 3:
                obs = observations.unsqueeze(0).unsqueeze(0)
                batch_size = 1
            else:
                obs = observations.unsqueeze(1)
                batch_size = observations.shape[0]

            obs = self._ensure_channel_first(obs)
            obs = self._normalize(obs)
            q_values, _, _, self.q_hidden = self.q_net(obs, self.q_hidden)

            q_values = q_values.squeeze(1)
            # Kept for eval diagnostics: magnitude of Q at deployment time
            self.last_q = q_values

            if deterministic:
                action = q_values.argmax(dim=1)
            else:
                # Per-row epsilon-greedy so parallel envs explore independently
                greedy = q_values.argmax(dim=1)
                rand_actions = torch.randint(
                    0, self.action_dim, (batch_size,), device=greedy.device
                )
                explore = torch.rand(batch_size, device=greedy.device) < self.epsilon
                action = torch.where(explore, rand_actions, greedy)
        self.q_net.train()
        return action

    def predict(self, observations, deterministic: bool = True):
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(observations, dtype=torch.float32).to(
                self.device_name
            )

        action = self.act(observations, deterministic=deterministic)
        return action.item() if action.numel() == 1 else action.cpu().numpy()

    def store_episode(self, episode):
        self.replay_buffer.append(episode)

    def store_transition(self, obs, action, reward, next_obs, terminated, truncated):
        pass

    def update_value(
        self,
        obs,
        actions,
        rewards,
        dones,
        masks,
        init_hidden=None,
        burn_in=16,
        train=False,
    ) -> dict:
        with self._autocast():
            h0 = self._make_hidden(init_hidden)
            if burn_in > 0:
                with torch.no_grad():
                    burn_obs = obs[:, :burn_in, :]
                    _, _, _, h_q = self.q_net(burn_obs, hidden=h0)
                    _, _, _, h_q_target = self.q_target(burn_obs, hidden=h0)
            else:
                h_q, h_q_target = h0, h0

            obs_active = obs[:, burn_in:, :]

            if train:
                _, v_full, _, _ = self.q_net(obs_active, hidden=h_q)
                current_v = v_full[:, :-1, :].squeeze(-1)

                with torch.no_grad():
                    _, v_target_full, _, _ = self.q_target(
                        obs_active, hidden=h_q_target
                    )
                    next_v = v_target_full[:, 1:, :].squeeze(-1)

                r_active = rewards[:, burn_in:]
                d_active = dones[:, burn_in:]
                m_active = masks[:, burn_in:]

                target_v = r_active + (1.0 - d_active) * self.gamma * next_v
                loss_v_unmasked = F.mse_loss(current_v, target_v, reduction="none")

                valid_steps = m_active.sum() + 1e-8
                loss_v = (loss_v_unmasked * m_active).sum() / valid_steps
            else:
                with torch.no_grad():
                    _, v_full, _, _ = self.q_net(obs_active, hidden=h_q)
                    current_v = v_full[:, :-1, :].squeeze(-1)

                    _, v_target_full, _, _ = self.q_target(
                        obs_active, hidden=h_q_target
                    )
                    next_v = v_target_full[:, 1:, :].squeeze(-1)

        if train:
            self.q_optimizer.zero_grad()
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
            self.q_optimizer.step()

            for param, target_param in zip(
                self.q_net.parameters(), self.q_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            loss_v_val = loss_v.item()
            self.epsilon = 0.01
        else:
            loss_v_val = 0.0

        return {
            "loss_v": loss_v_val,
            "v_mean": current_v.mean().item(),
            "current_v": current_v.detach(),
            "next_v": next_v.detach(),
        }

    def td_loss(
        self,
        obs,
        actions,
        rewards,
        dones,
        masks,
        init_hidden=None,
        burn_in=16,
        use_cql=True,
        n_step=1,
        is_weights=None,
    ):
        """Builds the (CQL-)TD loss without stepping the optimizer, so it can
        be combined with other terms into one backward pass. is_weights: (B,)
        PER importance-sampling corrections applied to the TD term. Returns
        (loss, aux) where aux includes per-step |TD error| ("td_abs", detached,
        active region only) for priority updates."""
        with self._autocast():
            with torch.no_grad():
                h0 = self._make_hidden(init_hidden)
                if burn_in > 0:
                    burn_obs = obs[:, :burn_in, :]
                    _, _, _, h_q = self.q_net(burn_obs, hidden=h0)
                    _, _, _, h_q_target = self.q_target(burn_obs, hidden=h0)
                else:
                    h_q, h_q_target = h0, h0

            obs_active = obs[:, burn_in:, :]
            a_active = actions[:, burn_in:]
            r_active = rewards[:, burn_in:]
            d_active = dones[:, burn_in:]
            m_active = masks[:, burn_in:]

            q_logits_full, v_full, adv_full, _ = self.q_net(obs_active, hidden=h_q)
            q_logits = q_logits_full[:, :-1, :]
            current_q = q_logits.gather(2, a_active.unsqueeze(-1)).squeeze(-1)
            current_v = v_full[:, :-1, :].squeeze(-1)

            with torch.no_grad():
                q_target_full, v_target_full, _, _ = self.q_target(
                    obs_active, hidden=h_q_target
                )
                # Double DQN: action selection by the online net, evaluation by the target net
                next_actions = q_logits_full[:, 1:, :].argmax(dim=2, keepdim=True)
                next_q = q_target_full[:, 1:, :].gather(2, next_actions).squeeze(-1)
                next_v = v_target_full[:, 1:, :].squeeze(-1)
                target_q = r_active + (1.0 - d_active) * self.gamma * next_q
                # Uncorrected n-step returns (Rainbow/R2D2 style). Each pass deepens
                # every position's return by one step via its right neighbor,
                # truncating at episode ends (dones) and at the window end (the
                # last column keeps its shallower return).
                for _ in range(n_step - 1):
                    shifted = torch.cat([target_q[:, 1:], target_q[:, -1:]], dim=1)
                    extended = r_active + (1.0 - d_active) * self.gamma * shifted
                    extended[:, -1] = target_q[:, -1]
                    target_q = extended

            q_td_loss_unmasked = F.mse_loss(current_q, target_q, reduction="none")
            if is_weights is not None:
                q_td_loss_unmasked = q_td_loss_unmasked * is_weights.unsqueeze(1)

            valid_steps = m_active.sum() + 1e-8
            q_loss = (q_td_loss_unmasked * m_active).sum() / valid_steps

            cql_loss = torch.tensor(0.0).to(self.device_name)
            if use_cql:
                cql_loss_unmasked = torch.logsumexp(q_logits, dim=2) - current_q
                cql_loss = (cql_loss_unmasked * m_active).sum() / valid_steps
                total_loss = q_loss + self.cql_alpha * cql_loss
            else:
                total_loss = q_loss

        aux = {
            "q_loss": q_loss.item(),
            "cql_loss": cql_loss.item(),
            "h_q": h_q,
            "current_v": current_v.detach(),
            "next_v": next_v.detach(),
            "td_abs": (current_q - target_q).abs().detach(),
            "m_active": m_active,
            # Grad-attached outputs over the active steps, so an imitation
            # term on the same batch can reuse this forward instead of
            # building a second activation graph (IMPALA graphs are ~GBs).
            "q_logits": q_logits,
            "adv_active": adv_full[:, :-1, :],
        }
        return total_loss, aux

    def soft_update_target(self):
        for param, target_param in zip(
            self.q_net.parameters(), self.q_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def update_td(
        self,
        obs,
        actions,
        rewards,
        dones,
        masks,
        init_hidden=None,
        burn_in=16,
        use_cql=True,
        n_step=1,
        is_weights=None,
    ) -> dict:
        total_loss, aux = self.td_loss(
            obs, actions, rewards, dones, masks,
            init_hidden=init_hidden, burn_in=burn_in,
            use_cql=use_cql, n_step=n_step, is_weights=is_weights,
        )

        self.q_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.q_optimizer.step()
        self.soft_update_target()

        return {
            "loss_td": total_loss.item(),
            **aux,
        }

    def supervised_loss(
        self,
        obs,
        actions,
        masks,
        init_hidden=None,
        burn_in=16,
        advantages=None,
        h_q=None,
        naive=False,
        bc_epsilon=None,
    ):
        """Builds the imitation (BC / AWBC) loss without stepping the
        optimizer, so it can be combined with other terms into one backward
        pass. Returns (loss, h_q)."""
        # bc_epsilon decouples the BC target entropy from the exploration schedule;
        # None preserves the coupled behavior (self.epsilon).
        eps = self.epsilon if bc_epsilon is None else bc_epsilon
        with self._autocast():
            if h_q is None:
                with torch.no_grad():
                    h0 = self._make_hidden(init_hidden)
                    if burn_in > 0:
                        burn_obs = obs[:, :burn_in, :]
                        _, _, _, h_q = self.q_net(burn_obs, hidden=h0)
                    else:
                        h_q = h0

            obs_active = obs[:, burn_in:-1, :]
            a_active = actions[:, burn_in:]
            m_active = masks[:, burn_in:]

            q_logits, _, adv_active, _ = self.q_net(obs_active, hidden=h_q)
            loss = self._imitation_loss_core(
                q_logits, adv_active, a_active, m_active,
                advantages=advantages, naive=naive, eps=eps,
            )
        return loss, h_q

    def _imitation_loss_core(self, q_logits, adv_active, a_active, m_active,
                             advantages=None, naive=False, eps=0.02):
        """Imitation loss from already-computed network outputs, so callers
        holding a live forward (e.g. the demo TD term) can reuse it."""
        act_dim = adv_active.shape[-1]
        adv_flat = adv_active.reshape(-1, act_dim)
        a_flat = a_active.reshape(-1)
        m_flat = m_active.reshape(-1)

        valid_indices = torch.nonzero(m_flat).squeeze(-1)
        if valid_indices.numel() == 0:
            return torch.tensor(0.0).to(self.device_name)

        valid_act = a_flat[valid_indices]
        if naive:
            q_logits_flat = q_logits.reshape(-1, act_dim)
            return F.cross_entropy(q_logits_flat[valid_indices], valid_act)
        if advantages is not None:
            adv_weight_flat = advantages.to(self.device_name).reshape(-1)
            return temperature_scaled_bc_loss(
                adv_flat[valid_indices],
                valid_act,
                epsilon=eps,
                weights=adv_weight_flat[valid_indices],
            )
        return temperature_scaled_bc_loss(
            adv_flat[valid_indices], valid_act, epsilon=eps
        )

    def update_supervised(
        self,
        obs,
        actions,
        masks,
        init_hidden=None,
        burn_in=16,
        anti=False,
        advantages=None,
        h_q=None,
        naive=False,
        bc_epsilon=None,
    ) -> dict:
        if anti:
            if h_q is None:
                with torch.no_grad():
                    h0 = self._make_hidden(init_hidden)
                    if burn_in > 0:
                        burn_obs = obs[:, :burn_in, :]
                        _, _, _, h_q = self.q_net(burn_obs, hidden=h0)
                    else:
                        h_q = h0

            obs_active = obs[:, burn_in:-1, :]
            a_active = actions[:, burn_in:]
            m_active = masks[:, burn_in:]

            q_logits, _, _, _ = self.q_net(obs_active, hidden=h_q)
            probs = F.softmax(q_logits, dim=2)
            bad_action_probs = probs.gather(2, a_active.unsqueeze(-1)).squeeze(-1)
            loss_unmasked = -torch.log(1 - bad_action_probs + 1e-8)

            valid_steps = m_active.sum() + 1e-8
            loss = (loss_unmasked * m_active).sum() / valid_steps

            self.q_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
            self.q_optimizer.step()

            return {"loss_supervised": loss.item()}

        loss, _ = self.supervised_loss(
            obs, actions, masks,
            init_hidden=init_hidden, burn_in=burn_in,
            advantages=advantages, h_q=h_q, naive=naive, bc_epsilon=bc_epsilon,
        )

        self.q_optimizer.zero_grad()
        if loss.requires_grad:
            (self.bc_weight * loss).backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
            self.q_optimizer.step()

        return {"loss_supervised": loss.item() if hasattr(loss, "item") else 0.0}

    def unified_update(
        self,
        online_batch=None,
        expert_batch=None,
        burn_in=16,
        online_n_step=1,
        expert_n_step=1,
        demo_td=False,
        use_cql=True,
        bc=False,
        awbc=False,
        bc_epsilon=None,
        is_weights=None,
        expert_advantages_fn=None,
    ) -> dict:
        """One combined loss over all active terms and a SINGLE backward /
        clip / optimizer step, so the demo TD, online TD, and imitation
        gradients stop thrashing each other through separate steps.

        online_batch / expert_batch: 6-tuples from sample_batch.
        is_weights: (B,) PER corrections for the online TD term.
        expert_advantages_fn: callable(aux) -> advantages for AWBC, given the
        demo TD aux dict (current_v / next_v); used only when awbc and demo_td.

        Each term's graph is backwarded as soon as it is complete, with
        gradients accumulating in .grad, then ONE clip / optimizer step at the
        end — numerically the same single update as backwarding the summed
        loss, but only one IMPALA activation graph (~GBs) is live at a time
        instead of two (the arm's expert + online graphs together OOM a 16GB
        card). The demo TD + imitation terms share a forward, so they
        backward together.
        """
        out = {}
        total_loss_val = 0.0
        did_backward = False
        self.q_optimizer.zero_grad()

        expert_term = None
        if expert_batch is not None and demo_td:
            loss_e, aux_e = self.td_loss(
                *expert_batch, burn_in=burn_in,
                use_cql=use_cql, n_step=expert_n_step,
            )
            expert_term = loss_e
            out["loss_td_expert"] = loss_e.item()
            out["cql_loss"] = aux_e["cql_loss"]
            out["h_q_expert"] = aux_e["h_q"]
            out["expert_aux"] = aux_e

        if expert_batch is not None and (bc or awbc) and self.bc_weight > 0:
            eps = self.epsilon if bc_epsilon is None else bc_epsilon
            if "expert_aux" in out:
                # Reuse the demo TD forward's outputs — its q_logits/adv cover
                # exactly the imitation term's active region, so no second
                # activation graph is built for the expert batch; for AWBC its
                # current_v/next_v also supply the advantage weights for free.
                aux_e = out["expert_aux"]
                advantages = (
                    expert_advantages_fn(aux_e)
                    if awbc and expert_advantages_fn is not None else None
                )
                loss_bc = self._imitation_loss_core(
                    aux_e["q_logits"], aux_e["adv_active"],
                    expert_batch[1][:, burn_in:], expert_batch[4][:, burn_in:],
                    advantages=advantages, eps=eps,
                )
            elif awbc and expert_advantages_fn is not None:
                # No demo TD forward this epoch: build the imitation forward
                # over the full active window (as td_loss does) so its own
                # (detached) V estimates feed the advantage weights — no
                # separate update_value rebuild of V(s); only next-state V
                # needs a no-grad target-net pass. Values are identical to
                # the old rebuild: same weights, same burn-in hiddens, and
                # the LSTM is causal so the extra trailing frame changes
                # nothing for the sliced steps. (In offline-only AWBC the fn
                # ignores this dict and trains V itself — see train_v.)
                obs_e, act_e = expert_batch[0], expert_batch[1]
                mask_e, hid_e = expert_batch[4], expert_batch[5]
                with self._autocast():
                    with torch.no_grad():
                        h0 = self._make_hidden(hid_e)
                        if burn_in > 0:
                            burn_obs = obs_e[:, :burn_in, :]
                            _, _, _, h_q = self.q_net(burn_obs, hidden=h0)
                            _, _, _, h_t = self.q_target(burn_obs, hidden=h0)
                        else:
                            h_q, h_t = h0, h0
                    obs_active = obs_e[:, burn_in:, :]
                    q_full, v_full, adv_full, _ = self.q_net(
                        obs_active, hidden=h_q
                    )
                    with torch.no_grad():
                        _, v_t_full, _, _ = self.q_target(obs_active, hidden=h_t)
                    advantages = expert_advantages_fn({
                        "current_v": v_full[:, :-1, :].squeeze(-1).detach(),
                        "next_v": v_t_full[:, 1:, :].squeeze(-1),
                    })
                    loss_bc = self._imitation_loss_core(
                        q_full[:, :-1, :], adv_full[:, :-1, :],
                        act_e[:, burn_in:], mask_e[:, burn_in:],
                        advantages=advantages, eps=eps,
                    )
            else:
                loss_bc, _ = self.supervised_loss(
                    expert_batch[0], expert_batch[1], expert_batch[4],
                    init_hidden=expert_batch[5], burn_in=burn_in,
                    bc_epsilon=bc_epsilon,
                )
            weighted_bc = loss_bc * self.bc_weight
            expert_term = weighted_bc if expert_term is None else expert_term + weighted_bc
            out["loss_supervised"] = loss_bc.item()

        if expert_term is not None and expert_term.requires_grad:
            total_loss_val += expert_term.item()
            expert_term.backward()
            did_backward = True

        if online_batch is not None:
            loss_o, aux_o = self.td_loss(
                *online_batch, burn_in=burn_in,
                use_cql=False, n_step=online_n_step, is_weights=is_weights,
            )
            out["loss_td_online"] = loss_o.item()
            out["td_abs_online"] = aux_o["td_abs"]
            out["m_active_online"] = aux_o["m_active"]
            if loss_o.requires_grad:
                total_loss_val += loss_o.item()
                loss_o.backward()
                did_backward = True

        if not did_backward:
            return out

        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.q_optimizer.step()
        self.soft_update_target()
        out["loss_total"] = total_loss_val
        return out

    def get_bc_loss(self, obs, actions, masks, init_hidden=None, burn_in=16) -> float:
        self.q_net.eval()
        with torch.no_grad():
            h0 = self._make_hidden(init_hidden)
            if burn_in > 0:
                burn_obs = obs[:, :burn_in, :]
                _, _, _, h_q = self.q_net(burn_obs, hidden=h0)
            else:
                h_q = h0

            obs_active = obs[:, burn_in:-1, :]
            a_active = actions[:, burn_in:]
            m_active = masks[:, burn_in:]

            _, _, adv_active, _ = self.q_net(obs_active, hidden=h_q)

            batch_size, seq_len, act_dim = adv_active.shape
            adv_flat = adv_active.reshape(-1, act_dim)
            a_flat = a_active.reshape(-1)
            m_flat = m_active.reshape(-1)

            valid_indices = torch.nonzero(m_flat).squeeze(-1)
            if valid_indices.numel() > 0:
                valid_adv = adv_flat[valid_indices]
                valid_act = a_flat[valid_indices]
                loss = F.cross_entropy(valid_adv, valid_act)
            else:
                loss = torch.tensor(0.0).to(self.device_name)

        self.q_net.train()
        return loss.item() if hasattr(loss, "item") else 0.0

    def get_logits(self, obs: torch.Tensor) -> torch.Tensor:
        if len(obs.shape) == 4:
            obs = obs.unsqueeze(1)
        obs = self._normalize(obs)
        logits, _, _, _ = self.q_net(obs)
        return logits

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        if len(obs.shape) == 4:
            obs = obs.unsqueeze(1)
        obs = self._normalize(obs)
        _, v, _, _ = self.q_net(obs)
        return v.squeeze(-1)

    def kl_update(self, obs: torch.Tensor, target_agent) -> dict:
        if len(obs.shape) == 4:
            obs = obs.unsqueeze(1)
        obs = self._normalize(obs)
        current_logits, _, _, _ = self.q_net(obs)
        with torch.no_grad():
            target_logits = target_agent.get_logits(obs)
            target_probs = F.softmax(target_logits, dim=-1)
        current_log_probs = F.log_softmax(current_logits, dim=-1)
        kl_loss = F.kl_div(
            current_log_probs.view(-1, self.action_dim),
            target_probs.view(-1, self.action_dim),
            reduction="batchmean",
        )
        self.q_optimizer.zero_grad()
        kl_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.q_optimizer.step()
        return {"loss_kl": kl_loss.item()}

    def to(self, device_name):
        self.device_name = device_name
        self.q_net.to(device_name)
        self.q_target.to(device_name)
        for state in self.q_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device_name)

    def sync_from(self, source_agent):
        with torch.no_grad():
            for p, src_p in zip(
                self.q_net.parameters(), source_agent.q_net.parameters()
            ):
                p.data.copy_(src_p.data)
            for p, src_p in zip(
                self.q_target.parameters(), source_agent.q_target.parameters()
            ):
                p.data.copy_(src_p.data)

    def _save_checkpoint(self, path):
        # Full training state so runs can be resumed, not just evaluated
        state = {
            "q_net": self.q_net.state_dict(),
            "q_target": self.q_target.state_dict(),
            "optimizer": self.q_optimizer.state_dict(),
        }
        torch.save(state, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device_name)
        if isinstance(checkpoint, dict) and "q_net" in checkpoint:
            self.q_net.load_state_dict(checkpoint["q_net"])
            if "q_target" in checkpoint:
                self.q_target.load_state_dict(checkpoint["q_target"])
            else:
                self.q_target = copy.deepcopy(self.q_net)
            if "optimizer" in checkpoint:
                self.q_optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            self.q_net.load_state_dict(checkpoint)
            self.q_target = copy.deepcopy(self.q_net)
