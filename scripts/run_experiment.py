from __future__ import annotations

import zipfile
from pathlib import Path

import clip
import gdown
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import ttest_ind
from tqdm.auto import tqdm

from src.clip_bias_geometry.geometry import build_embedding_views
from src.clip_bias_geometry.prompts import OCCUPATIONS, PROMPT_TEMPLATES, fill_prompt

SEED = 5329
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B/32"
AGE_GROUP = "30-39"
SAMPLES_PER_RACE_GENDER = 300

PROJECT_DIR = Path("artifacts")
DATA_DIR = PROJECT_DIR / "fairface"
CACHE_DIR = PROJECT_DIR / "cache"
RESULT_DIR = PROJECT_DIR / "results"
TABLE_DIR = RESULT_DIR / "tables"
for path in [DATA_DIR, CACHE_DIR, RESULT_DIR, TABLE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

IMG_ZIP_ID = "1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86"
TRAIN_LABEL_ID = "1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH"
IMG_ZIP = DATA_DIR / "fairface_data_img.zip"
IMG_ROOT = DATA_DIR / "fairface_data"
TRAIN_LABEL = DATA_DIR / "fairface_label_train.csv"


def prepare_fairface() -> pd.DataFrame:
    if not IMG_ZIP.exists():
        gdown.download(f"https://drive.google.com/uc?id={IMG_ZIP_ID}", str(IMG_ZIP), quiet=False)
    if not IMG_ROOT.exists():
        with zipfile.ZipFile(IMG_ZIP, "r") as zip_ref:
            zip_ref.extractall(IMG_ROOT)
    if not TRAIN_LABEL.exists():
        gdown.download(f"https://drive.google.com/uc?id={TRAIN_LABEL_ID}", str(TRAIN_LABEL), quiet=False)

    fairface = pd.read_csv(TRAIN_LABEL)
    age_df = fairface[fairface["age"] == AGE_GROUP].copy()
    age_df["image_path"] = age_df["file"].apply(lambda x: str(IMG_ROOT / x))
    counts = age_df.groupby(["race", "gender"]).size()
    if counts.min() < SAMPLES_PER_RACE_GENDER:
        raise ValueError("Not enough images in at least one race × gender group.")

    base = (
        age_df.groupby(["race", "gender"], group_keys=False)
        .sample(n=SAMPLES_PER_RACE_GENDER, random_state=SEED)
        .reset_index(drop=True)
    )
    base.to_csv(TABLE_DIR / "controlled_fairface_base_pool.csv", index=False)
    return base


def sample_n(group: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(group) < n:
        raise ValueError(f"Need {n} images but found {len(group)}")
    return group.sample(n=n, random_state=seed)


def build_data_settings(df: pd.DataFrame) -> dict[str, np.ndarray]:
    settings = {
        "balanced_50_50": [],
        "male_skewed_80_20": [],
        "female_skewed_20_80": [],
    }
    for race in sorted(df["race"].unique()):
        male = df[(df["race"] == race) & (df["gender"] == "Male")]
        female = df[(df["race"] == race) & (df["gender"] == "Female")]
        settings["balanced_50_50"] += [sample_n(male, 150, SEED + 1), sample_n(female, 150, SEED + 2)]
        settings["male_skewed_80_20"] += [sample_n(male, 240, SEED + 3), sample_n(female, 60, SEED + 4)]
        settings["female_skewed_20_80"] += [sample_n(male, 60, SEED + 5), sample_n(female, 240, SEED + 6)]
    return {name: pd.concat(parts).index.to_numpy() for name, parts in settings.items()}


@torch.no_grad()
def encode_images(model, preprocess, image_paths: list[str], batch_size: int = 64) -> np.ndarray:
    batches = []
    for start in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
        batch_paths = image_paths[start:start + batch_size]
        images = [preprocess(Image.open(path).convert("RGB")) for path in batch_paths]
        tensor = torch.stack(images).to(DEVICE)
        features = model.encode_image(tensor).float()
        features = features / features.norm(dim=-1, keepdim=True)
        batches.append(features.cpu().numpy())
    return np.vstack(batches)


@torch.no_grad()
def encode_texts(model, prompts: list[str]) -> np.ndarray:
    tokens = clip.tokenize(prompts, truncate=True).to(DEVICE)
    features = model.encode_text(tokens).float()
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()


def build_text_embeddings(model) -> tuple[pd.DataFrame, np.ndarray]:
    rows, vectors = [], []
    for prompt_type, template in PROMPT_TEMPLATES.items():
        prompts = [fill_prompt(template, job) for job in OCCUPATIONS]
        encoded = encode_texts(model, prompts)
        for job, prompt, vector in zip(OCCUPATIONS, prompts, encoded):
            rows.append({"prompt_type": prompt_type, "job": job, "prompt": prompt})
            vectors.append(vector)
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "encoded_prompt_table.csv", index=False)
    return table, np.vstack(vectors).astype(np.float64)


def compute_factorial_results(
    base_df: pd.DataFrame,
    data_settings: dict[str, np.ndarray],
    text_table: pd.DataFrame,
    embedding_views: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    text_lookup = {
        (row.prompt_type, row.job): idx
        for idx, row in text_table.iterrows()
    }
    rows = []
    for data_setting, indices in tqdm(data_settings.items(), desc="Data settings"):
        for prompt_type in PROMPT_TEMPLATES:
            for embedding_method, views in embedding_views.items():
                for job in OCCUPATIONS:
                    text_idx = text_lookup[(prompt_type, job)]
                    text_vector = views["text"][text_idx]
                    tmp = base_df.loc[indices].copy()
                    tmp["similarity"] = views["image"][indices] @ text_vector
                    male = tmp.loc[tmp["gender"] == "Male", "similarity"].to_numpy()
                    female = tmp.loc[tmp["gender"] == "Female", "similarity"].to_numpy()
                    signed_gap = male.mean() - female.mean()
                    rows.append({
                        "data_setting": data_setting,
                        "prompt_type": prompt_type,
                        "embedding_method": embedding_method,
                        "job": job,
                        "male_n": len(male),
                        "female_n": len(female),
                        "male_mean_similarity": male.mean(),
                        "female_mean_similarity": female.mean(),
                        "bias_male_minus_female": signed_gap,
                        "absolute_bias": abs(signed_gap),
                        "direction": "male_higher" if signed_gap > 0 else "female_higher" if signed_gap < 0 else "equal",
                        "welch_t_p_value": ttest_ind(male, female, equal_var=False).pvalue,
                    })
    return pd.DataFrame(rows)


def main() -> None:
    print("Device:", DEVICE)
    base_df = prepare_fairface()
    data_settings = build_data_settings(base_df)

    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    model.eval()

    cache_file = CACHE_DIR / "fairface_age_30_39_n300_image_embeddings_vit_b_32.npy"
    if cache_file.exists():
        image_embeddings = np.load(cache_file)
    else:
        image_embeddings = encode_images(model, preprocess, base_df["image_path"].tolist())
        np.save(cache_file, image_embeddings)

    text_table, text_embeddings = build_text_embeddings(model)
    embedding_views = build_embedding_views(
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        genders=base_df["gender"].to_numpy(),
    )
    results = compute_factorial_results(base_df, data_settings, text_table, embedding_views)
    output = TABLE_DIR / "factorial_absolute_bias_results_by_job.csv"
    results.to_csv(output, index=False)
    print(f"Saved {len(results)} rows to {output}")


if __name__ == "__main__":
    main()
