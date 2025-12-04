import os
from urllib.request import urlopen

# Split EDE versions out for flexible user requests, but both write to ede subdirectory.
COSMOPOWER_MODELS = {
    "ede-v2": [
        "growth-and-distances/DAZ_v2.npz",
        "growth-and-distances/HZ_v2.npz",
        "growth-and-distances/S8Z_v2.npz",
        "PK/PKL_v2.npz",
        "PK/PKNL_v2.npz",
    ],
    "ede-v1": [
        "growth-and-distances/DAZ_v1.npz",
        "growth-and-distances/HZ_v1.npz",
        "growth-and-distances/S8Z_v1.npz",
        "PK/PKL_v1.npz",
        "PK/PKNL_v1.npz",
    ],
    # all other standard models as before
    "lcdm": [
        "growth-and-distances/DAZ_v1.npz",
        "growth-and-distances/HZ_v1.npz",
        "growth-and-distances/S8Z_v1.npz",
        "PK/PKL_v1.npz",
        "PK/PKNL_v1.npz",
    ],
    "mnu": [
        "growth-and-distances/DAZ_mnu_v1.npz",
        "growth-and-distances/HZ_mnu_v1.npz",
        "growth-and-distances/S8Z_mnu_v1.npz",
        "PK/PKL_mnu_v1.npz",
        "PK/PKNL_mnu_v1.npz",
    ],
    "neff": [
        "growth-and-distances/DAZ_neff_v1.npz",
        "growth-and-distances/HZ_neff_v1.npz",
        "growth-and-distances/S8Z_neff_v1.npz",
        "PK/PKL_neff_v1.npz",
        "PK/PKNL_neff_v1.npz",
    ],
    "wcdm": [
        "growth-and-distances/DAZ_w_v1.npz",
        "growth-and-distances/HZ_w_v1.npz",
        "growth-and-distances/S8Z_w_v1.npz",
        "PK/PKL_w_v1.npz",
        "PK/PKNL_w_v1.npz",
    ],
    "mnu-3states": [
        "growth-and-distances/DAZ_v1.npz",
        "growth-and-distances/HZ_v1.npz",
        "growth-and-distances/S8Z_v1.npz",
        "PK/PKL_v1.npz",
        "PK/PKNL_v1.npz",
    ],
}


def get_default_data_path():
    return os.environ.get("HMFAST_EMULATOR_PATH", os.path.join(os.path.expanduser("~"), "hmfast_data"))


def download_emulators(target_dir=None, models="ede-v2", skip_existing=True):
    """
    Download emulator .npz files for specified cosmopower models into target_dir.

    - models: list of model names (e.g. ["ede-v1", "lcdm", ...]), or "all" to download every known model.
    - By default, only ede-v2 is downloaded.

    Other logic as previously described.
    """
    if target_dir is None:
        target_dir = os.path.join(os.path.expanduser("~"), "hmfast_data")
    os.makedirs(target_dir, exist_ok=True)

    valid_models = list(COSMOPOWER_MODELS.keys())

    # If user passed "all", download everything
    if models == "all" or models == ["all"]:
        models_to_fetch = valid_models
    elif models is None:
        models_to_fetch = ["ede-v2"]
    elif isinstance(models, str):
        models_to_fetch = [models]
    else:
        models_to_fetch = models

    # Validate models
    for m in models_to_fetch:
        if m not in valid_models:
            raise ValueError(f"Unknown model '{m}'. Available: {valid_models}")

    print(f"Downloading emulators to: {target_dir}")
    for model in models_to_fetch:
        subdir = "ede" if model.startswith("ede") else model
        print(f"Downloading cosmopower model: {model} → {subdir}/")
        for rel_path in COSMOPOWER_MODELS[model]:
            url = f"https://github.com/cosmopower-organization/{subdir}/raw/main/{rel_path}"
            local_dir = os.path.join(target_dir, subdir, os.path.dirname(rel_path))
            local_path = os.path.join(local_dir, os.path.basename(rel_path))
            os.makedirs(local_dir, exist_ok=True)
            if skip_existing and os.path.exists(local_path):
                print(f"  Already exists: {local_path} (skipped)")
                continue
            print(f"  Fetching {rel_path} ...")
            try:
                with urlopen(url) as resp, open(local_path, "wb") as out_file:
                    out_file.write(resp.read())
            except Exception as e:
                print(f"  *** Error downloading {rel_path}: {e}")
    print("Download complete.")
