"""Adaptive BID sampling. No weight, denoising, action, or episode-budget edits."""
import torch
from diffusion_policy.sampler.multi import bidirectional_sampler

SMALL = 6
FULL = 15
LOOKAHEAD = 4
AGREEMENT_PIXELS = 24.0
PRIOR_PIXELS = 32.0
MOTION_PIXELS = 20.0


class CachedPolicy:
    def __init__(self, original, predictions):
        self.original = original
        self.predictions = predictions

    def __getattr__(self, name):
        return getattr(self.original, name)

    def predict_action(self, obs):
        assert obs['obs'].shape[0] == self.predictions['action'].shape[0]
        return self.predictions


def candidates(policy, obs, count):
    b, oh, od = obs['obs'].shape
    expanded = {k: v[:, None].repeat(1, count, 1, 1).reshape(b * count, oh, od)
                for k, v in obs.items() if k != 'prior'}
    result = policy.predict_action(expanded)
    return {k: v.reshape(b, count, *v.shape[1:]) for k, v in result.items()}


def select(strong, weak, obs, prior, positive, negative):
    b, n = positive['action'].shape[:2]
    flatten = lambda values: {k: v.reshape(b * n, *v.shape[2:]) for k, v in values.items()}
    return bidirectional_sampler(CachedPolicy(strong, flatten(positive)),
                                 CachedPolicy(weak, flatten(negative)), obs, prior,
                                 num_sample=n, beta=0.5, num_mode=3)


class AdaptiveBID:
    """Choose budgets from prior confidence, invalidated by observed motion.

    Variable budgets are packed into one sampling batch per frozen policy.
    The gate uses generated plans and the policy's public keypoint input only.
    """
    def __init__(self):
        self.last_easy = None

    def step(self, strong, weak, obs, prior, active):
        motion = torch.linalg.vector_norm(
            (obs['obs'][:, -1, :18] - obs['obs'][:, -2, :18]).reshape(-1, 9, 2),
            dim=-1).mean(-1)
        easy = [False] * len(active) if self.last_easy is None else self.last_easy
        counts = [SMALL if easy[i] and float(motion[i]) <= MOTION_PIXELS else FULL
                  for i in range(len(active))]
        counts = [n if active[i] else SMALL for i, n in enumerate(counts)]
        repeats = torch.tensor(counts, device=obs['obs'].device)
        expanded = {k: v.repeat_interleave(repeats, dim=0) for k, v in obs.items()}
        positive = strong.predict_action(expanded)
        negative = weak.predict_action(expanded)
        pos_parts = {k: v.split(counts, dim=0) for k, v in positive.items()}
        neg_parts = {k: v.split(counts, dim=0) for k, v in negative.items()}
        outputs = []
        self.last_easy = []
        start = strong.n_obs_steps - 1
        for row, count in enumerate(counts):
            pos = {k: values[row][None] for k, values in pos_parts.items()}
            neg = {k: values[row][None] for k, values in neg_parts.items()}
            previous = None if prior is None else prior[row:row + 1]
            chosen = select(strong, weak, {k: v[row:row + 1] for k, v in obs.items()},
                            previous, pos, neg)
            outputs.append(chosen)
            path = pos['action_pred'][:, :, start:start + LOOKAHEAD]
            target = chosen['action_pred'][:, None, start:start + LOOKAHEAD]
            distances = torch.linalg.vector_norm(path - target, dim=-1).mean(-1)
            agreement = int((distances <= AGREEMENT_PIXELS).sum()) >= count // 3 + 1
            stable = True
            if previous is not None:
                overlap = min(LOOKAHEAD, previous.shape[1] - start)
                drift = torch.linalg.vector_norm(
                    chosen['action_pred'][:, start:start + overlap] -
                    previous[:, start:start + overlap], dim=-1).mean()
                stable = float(drift) <= PRIOR_PIXELS
            self.last_easy.append(agreement and stable)
        return {k: torch.cat([out[k] for out in outputs]) for k in outputs[0]}, counts
