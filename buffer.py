import torch
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler

# CyberRunner board dimensions (meters), used for mirror augmentation.
_BOARD_WIDTH = 0.276
_BOARD_HEIGHT = 0.231

# (flip_x, flip_y) configs for the 3 non-identity mirrors.
_MIRROR_CONFIGS = [(True, False), (False, True), (True, True)]


class Buffer:
    def __init__(self, config, mirror_augment=False):
        self.device = torch.device(config.device)
        self.storage_device = torch.device(config.storage_device)
        self.batch_size = int(config.batch_size)
        self.batch_length = int(config.batch_length)
        self.mirror_augment = mirror_augment
        self.num_eps = 0
        self._buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=config.max_size, device=self.storage_device, ndim=2),
            sampler=SliceSampler(
                num_slices=self.batch_size, end_key=None, traj_key="episode", truncated_key=None, strict_length=True
            ),
            prefetch=0,
            batch_size=self.batch_size * (self.batch_length + 1),  # +1 for context
        )

    def add_transition(self, data):
        if not self.mirror_augment:
            self._buffer.extend(data.unsqueeze(1))
            return

        env_num = data.shape[0]
        copies = [data]
        for i, (fx, fy) in enumerate(_MIRROR_CONFIGS, start=1):
            mirrored = self._mirror(data, fx, fy)
            mirrored["episode"] = data["episode"] + i * env_num
            copies.append(mirrored)
        # (env_num*4, ...) -> (env_num*4, 1, ...)
        self._buffer.extend(torch.cat(copies, dim=0).unsqueeze(1))

    @staticmethod
    def _mirror(data, flip_x, flip_y):
        """Create a mirrored copy of a batch of CyberRunner transitions."""
        data = data.clone()
        states = data["states"].clone()
        action = data["action"].clone()

        if flip_x:
            action[..., 0] *= -1  # alpha motor
            states[..., 0] *= -1  # alpha angle
            states[..., 2] = _BOARD_WIDTH - states[..., 2]  # ball x
            states[..., 4] *= -1  # vec_to_closest x
            states[..., 6] *= -1  # vec_to_next_wp x
            states[..., 8] *= -1  # vec_to_next_next_wp x
            if "image" in data:
                data["image"] = data["image"].flip(-2)  # flip width (HWC)

        if flip_y:
            action[..., 1] *= -1  # beta motor
            states[..., 1] *= -1  # beta angle
            states[..., 3] = _BOARD_HEIGHT - states[..., 3]  # ball y
            states[..., 5] *= -1  # vec_to_closest y
            states[..., 7] *= -1  # vec_to_next_wp y
            states[..., 9] *= -1  # vec_to_next_next_wp y
            if "image" in data:
                data["image"] = data["image"].flip(-3)  # flip height (HWC)

        data["states"] = states
        data["action"] = action
        return data

    def sample(self):
        sample_td, info = self._buffer.sample(return_info=True)
        # The sampler returns a flattened batch of length B*(T+1).
        # (B*(T+1), ...) -> (B, T+1, ...)
        sample_td = sample_td.view(-1, self.batch_length + 1)
        src_dev = sample_td.device
        if src_dev.type == "cpu" and self.device.type == "cuda":
            sample_td = sample_td.pin_memory().to(self.device, non_blocking=True)
        elif src_dev != self.device:
            sample_td = sample_td.to(self.device, non_blocking=True)
        # The initial ones are used only to extract the latent vector
        initial = (sample_td["stoch"][:, 0], sample_td["deter"][:, 0])
        data = sample_td[:, 1:]
        data.set_("action", sample_td["action"][:, :-1])  # action is 1 step back
        index = [ind.view(-1, self.batch_length + 1)[:, 1:] for ind in info["index"]]
        return data, index, initial

    def update(self, index, stoch, deter):
        # Flatten the data
        index = [ind.reshape(-1) for ind in index]
        # (B, T, S, K) -> (B*T, S, K)
        stoch = stoch.reshape(-1, *stoch.shape[2:])
        # (B, T, D) -> (B*T, D)
        deter = deter.reshape(-1, *deter.shape[2:])
        # In storage, the length is the first dimension, and the batch (number of environments) is the second dimension.
        self._buffer[index[1], index[0]].set_("stoch", stoch)
        self._buffer[index[1], index[0]].set_("deter", deter)

    def count(self):
        if self._buffer.storage.shape is None:
            return 0
        return self._buffer.storage.shape.numel()
