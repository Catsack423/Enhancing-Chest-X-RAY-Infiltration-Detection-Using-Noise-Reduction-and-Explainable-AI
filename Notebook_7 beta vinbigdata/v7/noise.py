"""Fixed, reproducible synthetic corruptions. Estimates are not detector noise truth."""
from hashlib import sha256
import json
import numpy as np

STATES = [{"id": "clean", "family": "clean", "level": 0}]
for family, values in (
    ("gaussian", [{"sigma": s} for s in (.02, .05, .10)]),
    ("poisson", [{"peak": p} for p in (500, 100, 30)]),
    ("poisson_gaussian", [{"peak": p, "sigma": s} for p, s in ((500, .01), (100, .03), (30, .05))]),
    ("salt_pepper", [{"amount": a} for a in (.005, .02, .05)]),
    ("speckle", [{"sigma": s} for s in (.05, .10, .20)]),
):
    STATES.extend({"id": f"{family}_N{i}", "family": family, "level": i, **params}
                  for i, params in enumerate(values, 1))
STATE_IDS = tuple(state["id"] for state in STATES)


def rng_for(seed, image_id, state, epoch=0, purpose="corruption"):
    identity = json.dumps([int(seed), str(image_id), str(state), int(epoch), purpose], separators=(",", ":"))
    words = np.frombuffer(sha256(identity.encode()).digest(), dtype="<u4")
    return np.random.default_rng(np.random.SeedSequence(words))


def training_state(seed, image_id, epoch):
    return STATE_IDS[int(rng_for(seed, image_id, "mixture", epoch, "choice").integers(len(STATES)))]


def corrupt(image, state_id="clean", *, seed=42, image_id="", epoch=0):
    """Input/output uint8; report corruption after clipping AND quantization."""
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError("Corruption requires a 2-D uint8 reference")
    state = next((s for s in STATES if s["id"] == state_id), None)
    if state is None:
        raise ValueError(f"Unknown corruption: {state_id}")
    x = image.astype(np.float32) / 255
    rng = rng_for(seed, image_id, state_id, epoch)
    family = state["family"]
    if family == "clean":
        y = x.copy()
    elif family == "gaussian":
        y = x + rng.normal(0, state["sigma"], x.shape)
    elif family in ("poisson", "poisson_gaussian"):
        y = rng.poisson(x * state["peak"]) / state["peak"]
        if family == "poisson_gaussian":
            y = y + rng.normal(0, state["sigma"], x.shape)
    elif family == "salt_pepper":
        y = x.copy()
        draws = rng.random(x.shape)
        y[draws < state["amount"] / 2] = 0
        y[draws >= 1 - state["amount"] / 2] = 1
    elif family == "speckle":
        y = x * (1 + rng.normal(0, state["sigma"], x.shape))
    result = np.rint(np.clip(y, 0, 1) * 255).astype(np.uint8)
    delta = result.astype(np.float32) / 255 - x
    return result, {"corruption_id": state_id, "family": family, "noise_level": state["level"],
                    "epoch": int(epoch), "clipping_fraction": float(np.mean((y < 0) | (y > 1))),
                    "noise_mean": float(delta.mean()), "noise_std": float(delta.std()),
                    "noise_rmse": float(np.sqrt(np.mean(delta ** 2)))}
