"""Run simulator experiments (CPU) and detector training (GPU) on Modal.

One-time setup (from the project folder). On Windows, if `modal` is not on
PATH use `python -m modal ...`; in Git Bash run `export MSYS_NO_PATHCONV=1`
first, or volume paths like /Flood_Dataset.zip get rewritten (PowerShell is fine).
    pip install modal && modal setup
    modal volume create bflood-data
    modal volume put bflood-data data/Flood_Dataset.zip /Flood_Dataset.zip
    modal volume put bflood-data data/flooded_cars.zip /flooded_cars.zip
    modal volume put bflood-data data/AlleyFloodnet.zip /AlleyFloodnet.zip
    modal volume put bflood-data data/fisheye8k_sample.zip /fisheye8k_sample.zip
    modal volume put bflood-data data/processed /processed
    modal volume put bflood-data data/grok /grok            # once Grok images exist
(Re-upload data/processed after rebuilding the manifest: add --force.)

Simulator:  modal run modal_app.py::main --config exp9_worstcase.yaml --extra "--bank detector_base --quick"
            modal run --detach modal_app.py::main --config exp1_main.yaml --extra "--bank detector_base"
            (results: modal volume get bflood-results exp1_main results/)
Detector:   modal run modal_app.py::train --config detector_smoke.yaml
            modal run --detach modal_app.py::train --config detector_base.yaml
Results:    modal volume get bflood-results detector/<name> results/
"""
import modal

app = modal.App("bflood")
results = modal.Volume.from_name("bflood-results", create_if_missing=True)
data = modal.Volume.from_name("bflood-data", create_if_missing=True)

sim_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "scipy", "networkx", "pyyaml")
    .add_local_dir("src", "/root/src")
    .add_local_dir("configs", "/root/configs")
    .add_local_dir("scripts", "/root/scripts")
)

det_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "opencv-python-headless", "numpy",
                 "scipy", "pillow", "pyyaml", "networkx")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", "/root/src")
    .add_local_dir("configs", "/root/configs")
)


@app.function(image=sim_image, cpu=8, timeout=3 * 3600, volumes={"/root/results": results})
def run_config(config_name: str, extra: str = "") -> str:
    """extra: flags for run_experiment.py, e.g. "--bank detector_base",
    "--placeholder" or "--quick". The belief bank of detector run <b> is read
    from the results volume at detector/<b>/ensemble/belief_bank.npz."""
    import shlex
    import subprocess
    import sys

    subprocess.run([sys.executable, "/root/scripts/run_experiment.py",
                    f"/root/configs/{config_name}", "--workers", "8", *shlex.split(extra)],
                   check=True, cwd="/root")
    results.commit()
    return f"done: {config_name} {extra}"


@app.function(image=det_image, gpu="L4", cpu=8, memory=16384, timeout=8 * 3600,
              volumes={"/data": data, "/root/results": results})
def train_detector(config_name: str) -> dict:
    from detector.train import run

    met = run(f"/root/configs/{config_name}", data_dir="/data", out_root="/root/results/detector")
    results.commit()
    return {k: v for k, v in met.items() if not isinstance(v, dict)}


@app.function(image=det_image, cpu=8, memory=16384, timeout=3600,
              volumes={"/data": data, "/root/results": results})
def adverse_bank(run: str = "detector_base") -> str:
    """exp13 condition model: the run's ensemble on its belief-bank images
    under the fixed adverse transform, CPU (src/detector/adverse_bank.py).
    Writes detector/<run>/ensemble/belief_bank_adverse.npz on the volume."""
    from detector.adverse_bank import run as make

    md = make("/data", f"/root/results/detector/{run}")
    results.commit()
    return md


@app.local_entrypoint()
def adverse(run: str = "detector_base"):
    print(adverse_bank.remote(run))


@app.local_entrypoint()
def main(config: str = "exp9_worstcase.yaml", extra: str = "--bank detector_base"):
    print(run_config.remote(config, extra))


@app.local_entrypoint()
def train(config: str = "detector_smoke.yaml"):
    print(train_detector.remote(config))
