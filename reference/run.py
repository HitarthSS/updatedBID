"""Paired BID experiment using native frozen policies, sampler, and Push-T."""
import os
os.environ.update(WANDB_MODE='disabled', WANDB_DISABLED='true', SDL_VIDEODRIVER='dummy',
                  PYGAME_HIDE_SUPPORT_PROMPT='1', OMP_NUM_THREADS='2')
import collections
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'upstream'))
import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.sampler.condition import NoiseGenerator
from diffusion_policy.sampler.multi import bidirectional_sampler
import my_harness as mine

torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
OmegaConf.register_new_resolver('eval', eval, replace=True)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def state_sha(policy):
    h = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load(epoch):
    path = next((ROOT / 'ckpt/pusht').glob('*' + epoch + '*'))
    payload = torch.load(path, map_location='cpu', pickle_module=dill, weights_only=False)
    cfg = payload['cfg']
    policy = hydra.utils.instantiate(cfg.policy)
    key = 'ema_model' if cfg.training.use_ema else 'model'
    policy.load_state_dict(payload['state_dicts'][key], strict=True)
    policy.requires_grad_(False).eval()
    policy.n_action_steps = 1
    metadata = dict(path=str(path), sha256=sha(path), state_dict=key,
                    policy_state_sha256=state_sha(policy),
                    config=OmegaConf.to_container(cfg, resolve=True))
    del payload
    policy.to('cuda:0')
    assert policy.horizon == 16 and policy.n_obs_steps == 2
    assert policy.num_inference_steps == 100
    return policy, metadata


class AuditEnv(PushTKeypointsEnv):
    def reset(self):
        self.native_success = False
        return super().reset()

    def step(self, action):
        result = super().step(action)
        self.native_success = self.native_success or bool(result[2])
        return result


def summary():
    pairs = []
    for seed in range(20100, 20220):
        paths = [ROOT / 'runs' / arm / str(seed) / 'result.json' for arm in ('authors', 'mine')]
        if not all(p.exists() for p in paths):
            continue
        a, b = [json.loads(p.read_text()) for p in paths]
        for key in ('initial_obs_sha256', 'initial_image_sha256', 'initial_state_sha256',
                    'noise_seed', 'policy_seed', 'max_steps'):
            if a[key] != b[key]:
                raise RuntimeError(f'Pair mismatch {seed}: {key}')
        pairs.append((a, b))
    out = dict(completed_pairs=len(pairs), planned_pairs=120, planned_episodes=240,
               primary_metric='mean episode maximum normalized overlap reward',
               reproduction='Single released stochastic Push-T condition; reconstructed RNG/batching/runtime, not exact paper aggregate')
    for i, arm in enumerate(('authors', 'mine')):
        rows = [p[i] for p in pairs]
        out[arm] = dict(episodes=len(rows), successes=sum(r['success'] for r in rows),
                        mean_score=float(np.mean([r['max_reward'] for r in rows])) if rows else None,
                        success_percent=100 * np.mean([r['success'] for r in rows]) if rows else None,
                        physical_steps=sum(r['physical_steps'] for r in rows),
                        candidate_trajectories=sum(r['candidate_trajectories'] for r in rows),
                        amortized_inference_seconds=sum(r['inference_seconds_amortized'] for r in rows),
                        horizon_violations=sum(r['horizon_violations'] for r in rows),
                        paid_api_calls=0, vlm_calls=0, vlm_tokens=0)
    if pairs:
        delta = np.array([b['max_reward'] - a['max_reward'] for a, b in pairs])
        rng = np.random.RandomState(4813)
        boot = delta[rng.randint(len(delta), size=(10000, len(delta)))].mean(1)
        out['paired'] = dict(score_difference=float(delta.mean()),
                             score_difference_bootstrap_95=np.percentile(boot, [2.5, 97.5]).tolist(),
                             rescued=sum(not a['success'] and b['success'] for a, b in pairs),
                             harmed=sum(a['success'] and not b['success'] for a, b in pairs))
    atomic(ROOT / 'SUMMARY.json', out)
    return out


def batch(strong, weak, seeds, arm, phase, kp, cap):
    assert cap == 300
    envs = []
    records = []
    observations = []
    policy_seed = seeds[0] + 700000
    random.seed(policy_seed); np.random.seed(policy_seed); torch.manual_seed(policy_seed)
    for seed in seeds:
        folder = ROOT / phase / arm / str(seed)
        if folder.exists():raise RuntimeError('Refusing to overwrite a partial or completed episode: '+str(folder))
        folder.mkdir(parents=True, exist_ok=False)
        raw = AuditEnv(legacy=True, keypoint_visible_rate=1.0, agent_keypoints=False,
                       render_action=False, **kp)
        env = MultiStepWrapper(raw, n_obs_steps=2, n_action_steps=1, max_episode_steps=cap)
        env.seed(seed)
        obs = env.reset()
        rgb = env.render(mode='rgb_array')
        Image.fromarray(rgb).save(folder / 'initial.png')
        state = np.array([*raw.agent.position, *raw.block.position, raw.block.angle,
                          *raw.agent.velocity, *raw.block.velocity, raw.block.angular_velocity])
        records.append(dict(seed=seed, arm=arm, phase=phase, max_steps=cap,
                            initial_obs_sha256=array_sha(obs), initial_image_sha256=array_sha(rgb),
                            initial_state_sha256=array_sha(state), noise_seed=seed + 900000,
                            policy_seed=policy_seed, candidate_trajectories=0, decisions=0,
                            small_decisions=0, full_decisions=0, inference_seconds_amortized=0.0,
                            simulator_seconds=0.0, paid_api_calls=0, vlm_calls=0, vlm_tokens=0))
        prior_states=json.loads((ROOT/'PRIOR_INITIAL_STATES.json').read_text())
        assert records[-1]['initial_state_sha256'] not in prior_states,'Start overlaps a prior executed cohort'
        peer=ROOT/phase/('mine' if arm=='authors' else 'authors')/str(seed)/'result.json'
        if peer.exists():
            previous=json.loads(peer.read_text())
            for key in ['initial_obs_sha256','initial_image_sha256','initial_state_sha256','noise_seed','policy_seed','max_steps']:
                assert previous[key]==records[-1][key],'Pair mismatch before actions: '+key
        envs.append(env); observations.append(obs)
    # Independent episode-local noise generators use the released formula.
    noises = [NoiseGenerator(1.0) for _ in seeds]
    noise_states = [np.random.RandomState(s + 900000).get_state() for s in seeds]
    strong.reset(); weak.reset()
    prior = None
    adaptive = mine.AdaptiveBID()
    active = [True] * len(seeds)
    batch_inference = 0.0
    started = time.monotonic()
    trace = []
    for step in range(8 if phase=='smoke' else cap):
        observation = np.stack(observations)
        obs = {'obs': torch.from_numpy(observation[..., :20].astype(np.float32)).to('cuda:0'),
               'obs_mask': torch.from_numpy(observation[..., 20:] > .5).to('cuda:0')}
        torch.cuda.synchronize(); t = time.monotonic()
        with torch.no_grad():
            if arm == 'authors':
                result = bidirectional_sampler(strong, weak, obs, prior, 15, 0.5, 3)
                counts = [15] * len(seeds)
            else:
                result, counts = adaptive.step(strong, weak, obs, prior, active)
        torch.cuda.synchronize(); elapsed = time.monotonic() - t
        batch_inference += elapsed
        prior = result['action_pred'][:, 1:]
        action = result['action'].detach().cpu().numpy().copy()
        prediction = result['action_pred'].detach().cpu().numpy()
        enabled_count = sum(active)
        trace.append(dict(step=step, active=list(active), candidates_per_policy=counts,
                          inference_seconds=elapsed))
        for i, env in enumerate(envs):
            if not active[i]:
                continue
            saved = np.random.get_state(); np.random.set_state(noise_states[i])
            disturbance = noises[i].step(prediction[i:i + 1])[0, :1]
            noise_states[i] = np.random.get_state(); np.random.set_state(saved)
            # Match the release's float32 in-place addition before physics.
            action[i] += disturbance
            applied = action[i]
            if not np.isfinite(applied).all():
                raise RuntimeError('Non-finite action')
            t = time.monotonic()
            observations[i], reward, done, info = env.step(applied)
            records[i]['simulator_seconds'] += time.monotonic() - t
            records[i]['candidate_trajectories'] += 2 * counts[i]
            records[i]['decisions'] += 1
            records[i]['small_decisions'] += counts[i] == 6
            records[i]['full_decisions'] += counts[i] == 15
            records[i]['inference_seconds_amortized'] += elapsed / enabled_count
            active[i] = not bool(done)
            if len(env.reward) > cap:
                raise RuntimeError('Physical horizon exceeded')
        atomic(ROOT / 'live.json', dict(time=time.time(), phase=phase, arm=arm,
               seeds=seeds, step=step + 1, active=active, batch_inference_seconds=batch_inference))
        if not any(active):
            break
    wall = time.monotonic() - started
    for i, (env, record) in enumerate(zip(envs, records)):
        record.update(success=bool(env.env.native_success), max_reward=float(max(env.reward)),
                      diagnostic=phase=='smoke', smoke_limit=8 if phase=='smoke' else None,
                      physical_steps=len(env.reward), horizon_violations=int(len(env.reward) > cap),
                      group_wall_seconds=wall, group_inference_seconds=batch_inference,
                      timing='GPU time amortized over active episodes; same batch size, shared GPU; includes padded-row compute')
        atomic(ROOT / phase / arm / str(seeds[i]) / 'result.json', record)
        env.close()
    atomic(ROOT / phase / f'{arm}-{seeds[0]}-trace.json', trace)
    return records


def parity(strong, weak):
    # Same cached predictions through the native and cached selection pathways.
    class Fake:
        n_obs_steps = 2
        n_action_steps = 1
        horizon = 16
    original = Fake()
    obs = {'obs': torch.zeros(2, 2, 20, device='cuda:0'),
           'obs_mask': torch.ones(2, 2, 20, device='cuda:0', dtype=torch.bool)}
    generator = torch.Generator(device='cuda:0').manual_seed(7102)
    checks = []
    for n in (6, 15):
        pos = {'action_pred': torch.randn(2, n, 16, 2, generator=generator, device='cuda:0')}
        neg = {'action_pred': torch.randn(2, n, 16, 2, generator=generator, device='cuda:0')}
        pos['action'] = pos['action_pred'][:, :, 1:2]
        neg['action'] = neg['action_pred'][:, :, 1:2]
        for prior in (None, torch.randn(2, 15, 2, generator=generator, device='cuda:0')):
            flat = lambda values: {k: v.reshape(2 * n, *v.shape[2:]) for k, v in values.items()}
            a = bidirectional_sampler(mine.CachedPolicy(original, flat(pos)),
                                      mine.CachedPolicy(original, flat(neg)), obs, prior, n, .5, 3)
            b = mine.select(original, original, obs, prior, pos, neg)
            assert all(torch.equal(a[k], b[k]) for k in a)
            checks.append(dict(candidates=n, prior=prior is not None, bitwise_equal=True))
    atomic(ROOT / 'SELECTOR_PARITY.json', dict(status='PASS', checks=checks))
    class CountingPolicy(Fake):
        def __init__(self):
            self.calls = []
        def predict_action(self, values):
            size = values['obs'].shape[0]
            self.calls.append(size)
            pred = torch.arange(size * 16 * 2, device='cuda:0', dtype=torch.float32).reshape(size, 16, 2) / 100
            return {'action_pred': pred, 'action': pred[:, 1:2]}
    p, q = CountingPolicy(), CountingPolicy()
    adaptive = mine.AdaptiveBID()
    _, counts = adaptive.step(p, q, obs, None, [True, True])
    assert counts == [15, 15] and p.calls == [30] and q.calls == [30]
    adaptive.last_easy = [True, False]
    _, counts = adaptive.step(p, q, obs, None, [True, True])
    assert counts == [6, 15] and p.calls == [30, 21] and q.calls == [30, 21]
    atomic(ROOT / 'BUDGET_VALIDATION.json', dict(status='PASS', first_counts=[15, 15],
           mixed_counts=counts, one_sampling_batch_per_policy=True, max_candidates_per_policy=15))


def freeze(model_metadata):
    files = [ROOT/name for name in ['run.py','my_harness.py','PROTOCOL.md','PARENT_FROZEN.json','PRIOR_INITIAL_STATES.json','READINESS.json','SELECTOR_PARITY.json','BUDGET_VALIDATION.json']]
    files += list((ROOT / 'upstream/diffusion_policy/sampler').glob('*.py'))
    files += list((ROOT / 'upstream/diffusion_policy/policy').glob('*.py'))
    files += list((ROOT / 'upstream/diffusion_policy/model').rglob('*.py'))
    files += list((ROOT / 'upstream/diffusion_policy/env/pusht').glob('*.py'))
    files += [ROOT / 'upstream/diffusion_policy/gym_util/multistep_wrapper.py']
    record = dict(time=time.time(), files={str(p.relative_to(ROOT)): sha(p) for p in files},
                  upstream_revision=subprocess.check_output(['git', '-C', str(ROOT / 'upstream'), 'rev-parse', 'HEAD'], text=True).strip(),
                  checkpoints=model_metadata,
                  python=sys.version, packages={k: importlib.metadata.version(k) for k in
                  ['torch', 'diffusers', 'numpy', 'gym', 'pymunk', 'hydra-core', 'pygame', 'shapely']},
                  hardware=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total', '--format=csv,noheader'], text=True),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  test_seeds=list(range(20100, 20220)), smoke_seeds=list(range(60100, 60104)), smoke_actions=8,
                  batch_size=4, episode_cap=300, noise=1.0, n_action_steps=1,
                  selection='Fresh predeclared rerun seeds20100..20219; original adaptive design unchanged; not the released seed0 cohort')
    assert not (ROOT / 'FROZEN.json').exists(), 'Never replace frozen design'
    atomic(ROOT / 'FROZEN.json', record)
    return record


def verify(record):
    for file, digest in record['files'].items():
        if sha(ROOT / file) != digest:
            raise RuntimeError('Frozen source changed: ' + file)


def main():
    lock=(ROOT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic(ROOT/'status.json',dict(state='loading',pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.18)
    strong,a=load('0150');weak,b=load('0050')
    parent=json.loads((ROOT/'PARENT_FROZEN.json').read_text())
    for label,meta in [('strong',a),('weak',b)]:
        assert meta['sha256']==parent['checkpoints'][label]['sha256']
        assert meta['policy_state_sha256']==parent['checkpoints'][label]['policy_state_sha256']
    assert sha(ROOT/'my_harness.py')==parent['files']['my_harness.py']
    assert a['config']['task']['env_runner']['max_steps']==300
    assert a['config']['task']['env_runner']['legacy_test'] is True
    kp=PushTKeypointsEnv.genenerate_keypoint_manager_params()
    parity(strong,weak)
    sources={p:sha(ROOT/p) for p in ['run.py','my_harness.py','PROTOCOL.md','PRIOR_INITIAL_STATES.json','PARENT_FROZEN.json']}
    if '--evaluate' not in sys.argv:
        assert not (ROOT/'SMOKE_FROZEN.json').exists()
        atomic(ROOT/'SMOKE_FROZEN.json',dict(sources=sources,physical_action_limit=8,seeds=list(range(60100,60104)),diagnostic_only=True))
        records={arm:batch(strong,weak,list(range(60100,60104)),arm,'smoke',kp,300) for arm in ['authors','mine']}
        for first,second in zip(records['authors'],records['mine']):
            for key in ['initial_obs_sha256','initial_image_sha256','initial_state_sha256','noise_seed','policy_seed','max_steps']:
                assert first[key]==second[key]
            assert first['physical_steps']<=8 and second['physical_steps']<=8
        assert state_sha(strong)==a['policy_state_sha256'] and state_sha(weak)==b['policy_state_sha256']
        assert sources=={p:sha(ROOT/p) for p in sources}
        atomic(ROOT/'SMOKE_RECORDS.json',records)
        atomic(ROOT/'READINESS.json',dict(status='PASS',sources=sources,weights_unchanged=True,paired_initial_states=True,
            physical_episodes=8,maximum_smoke_actions=64,excluded_from_results=True,paid_api_calls=0))
        atomic(ROOT/'status.json',dict(state='smoke_complete',pid=os.getpid(),weights_unchanged=True))
        return
    ready=json.loads((ROOT/'READINESS.json').read_text())
    assert ready['status']=='PASS' and ready['sources']==sources
    assert not (ROOT/'LAST_ERROR.json').exists()
    frozen=freeze(dict(strong=a,weak=b))
    for group_index,start in enumerate(range(20100,20220,4)):
        seeds=list(range(start,start+4))
        order=['authors','mine'] if group_index%2==0 else ['mine','authors']
        for arm in order:
            verify(frozen)
            atomic(ROOT/'status.json',dict(state='running',pid=os.getpid(),arm=arm,seeds=seeds,
                completed_pairs=summary()['completed_pairs'],planned_pairs=120,planned_episodes=240,updated_unix=time.time()))
            batch(strong,weak,seeds,arm,'runs',kp,300)
        print(json.dumps(summary()),flush=True)
    verify(frozen)
    assert state_sha(strong)==a['policy_state_sha256'] and state_sha(weak)==b['policy_state_sha256']
    result=summary();assert result['completed_pairs']==120
    for arm in ['authors','mine']:assert result[arm]['episodes']==120 and result[arm]['horizon_violations']==0
    atomic(ROOT/'FINAL_WEIGHTS.json',dict(status='PASS',frozen_policy_weights_unchanged=True))
    atomic(ROOT/'status.json',dict(state='complete',completed_pairs=120,completed_episodes=240,finished_unix=time.time(),pid=os.getpid()))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        atomic(ROOT / 'LAST_ERROR.json', dict(time=time.time(), error=traceback.format_exc()))
        raise
