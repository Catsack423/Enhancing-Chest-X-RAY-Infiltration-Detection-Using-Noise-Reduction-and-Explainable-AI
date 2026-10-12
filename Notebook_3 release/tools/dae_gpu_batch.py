"""Measure DAE training memory/throughput, then open a distinct batch-size run."""
import gc
import json
import time
import re
from pathlib import Path

def choose_batch(rows, budget_bytes):
    eligible = [r for r in rows if r['peak_bytes'] <= budget_bytes and r['images_per_second'] > 0]
    if not eligible:
        raise RuntimeError('No DAE batch fits the GPU budget; free GPU memory or restart the Python session')
    fastest = max(r['images_per_second'] for r in eligible)
    # Prefer a larger batch when speed is within 5% of the best measured result.
    return max((r for r in eligible if r['images_per_second'] >= .95 * fastest), key=lambda r:r['batch_size'])

def benchmark_dae(config, fraction=.90, candidates=(16,32,48,56,64,72,80,88,96,112,128,160,192,256), steps=3):
    import torch
    from shared.dae_model import DAE
    from shared.training import capture_rng_state, restore_rng_state, _autocast, _scaler, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError('Select a CUDA GPU runtime first')
    if not .5 <= fraction <= .95 or steps < 2:
        raise ValueError('Use a VRAM budget of 50..95% and at least two timed steps')
    saved = capture_rng_state()
    flags = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
             torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    settings = config['dae']
    amp = settings.get('amp', True)
    size = config['preprocessing']['working_size']
    rows = []
    device = torch.cuda.current_device()
    torch.cuda.empty_cache()
    if torch.cuda.memory_allocated(device) > 1024**3:
        raise RuntimeError('Old GPU model is still live. Restart Python session on this VM, rerun setup/preparation, then tune batch; keep dataset/cache.')
    free, total = torch.cuda.mem_get_info(device)
    external = max(0, total-free-torch.cuda.memory_reserved(device))
    budget = int(total*fraction)
    print(f'GPU {torch.cuda.get_device_name(device)} | total {total/1024**3:.1f} GiB | budget {fraction:.0%} | AMP={amp}', flush=True)
    try:
        set_seed(42)  # Match the deterministic backend used by real training.
        for batch in candidates:
            model = optimizer = scaler = clean = noisy = prediction = loss = None
            print(f'DAE batch probe {batch}: forward + backward + Adam; warmup then {steps} timed steps', flush=True)
            try:
                model = DAE(base_channels=settings.get('base_channels',32), residual=True).cuda().train()
                optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
                scaler = _scaler('cuda', amp)
                clean = torch.rand(batch,1,size,size,device='cuda')
                noisy = (clean + .05*torch.randn_like(clean)).clamp(0,1)
                def step():
                    optimizer.zero_grad(set_to_none=True)
                    with _autocast('cuda',amp):
                        prediction = model(noisy)
                        loss = torch.nn.functional.mse_loss(prediction,clean) + settings.get('l1_weight',.2)*torch.nn.functional.l1_loss(prediction,clean)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                torch.cuda.reset_peak_memory_stats(device)
                step()  # Includes Adam state allocation in peak VRAM.
                for _ in range(8):
                    if optimizer.state:
                        break
                    step()  # Recover initial AMP overflows before measuring Adam memory.
                if not optimizer.state:
                    raise RuntimeError('AMP skipped every probe update; cannot measure full training memory safely')
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                for _ in range(steps):
                    step()
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter()-started
                peak = torch.cuda.max_memory_reserved(device)+external
                row = {'batch_size':int(batch),'peak_bytes':int(peak),
                       'images_per_second':batch*steps/elapsed,'step_seconds':elapsed/steps}
                rows.append(row)
                print(f'  batch={batch} | peak {peak/1024**3:.2f}/{total/1024**3:.2f} GiB | {row["images_per_second"]:.1f} images/s', flush=True)
                if peak > budget:
                    print('  above budget; stop growing batch', flush=True)
                    break
            except torch.cuda.OutOfMemoryError:
                print(f'  batch={batch}: CUDA OOM; keep earlier safe candidates', flush=True)
                break
            finally:
                model = optimizer = scaler = clean = noisy = prediction = loss = None
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        restore_rng_state(saved)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = flags[:2]
        torch.use_deterministic_algorithms(flags[2], warn_only=flags[3])
    best = choose_batch(rows,budget)
    print(f'CHOSEN batch={best["batch_size"]} | fastest safe throughput (larger batch within 5%)', flush=True)
    return {'batch_size':best['batch_size'],'vram_budget_fraction':fraction,
            'gpu':torch.cuda.get_device_name(device),'total_vram_bytes':total,'measurements':rows,
            'scope':'synthetic GPU training steps; not end-to-end epoch timing or research results'}

def open_batch_runs(root, previous_runs, result):
    from shared.experiment import Experiment, digest, write_json
    from shared.data import prepare_manifests
    from shared.runtime_cache import RuntimeCache
    batch = int(result['batch_size'])
    if batch < 1:
        raise ValueError('Batch must be positive')
    new_runs = {}
    for seed, (old, _) in previous_runs.items():
        name = re.sub(r'_dae_b\d+_gpu$', '', old.base_output_root.name)
        base = old.base_output_root.with_name(f'{name}_dae_b{batch:03d}_gpu')
        if base.resolve() == old.base_output_root.resolve() and old.config['dae']['batch_size'] != batch:
            raise ValueError('GPU batch experiment must use a separate output root')
        if old.config['run_mode'] != 'full':
            raise ValueError('GPU batch tuning is for full runs; keep SMOKE=False')
        exp = Experiment.open(root,old.data_root,base,seed=seed,cache_root=old.cache_root)
        exp.config['dae']['batch_size'] = batch
        exp.staging_root = exp.cache_root/'staging'/digest(exp.config)/f'seed_{seed}'
        exp.staging_root.mkdir(parents=True,exist_ok=True)
        frames = prepare_manifests(exp)
        exp.cache = RuntimeCache(exp).prepare(frames)
        exp.lock()
        write_json(exp.artifacts/'gpu_batch_probe.json',result)
        new_runs[seed] = (exp,frames)
        print(f'GPU batch run: seed={seed} DAE batch={batch} | {exp.output_root}',flush=True)
    return new_runs
