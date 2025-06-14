from logging import Logger
from pathlib import Path
from typing import Optional, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


def load_faiss_index(faiss_index_path: str) -> faiss.Index:
    return faiss.read_index(faiss_index_path)


def save_faiss_index(faiss_index: faiss.Index, faiss_index_path: str) -> None:
    faiss.write_index(faiss_index, faiss_index_path)


def generate_embeddings(
    dataframe: pd.DataFrame,
    index_column: str,
    text_column: str,
    model_path: str,
    model_name: str,
    logger: Optional[Logger] = None,
    batch_size: int = 32,
    max_length: int = 128,
    pooling_method: str = "mean",
    print_iter: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    local_path = Path(model_path).absolute()

    if local_path.exists() and local_path.is_dir():
        model_path = str(local_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    embeddings = []

    n = len(dataframe) // batch_size
    for i in tqdm(range(0, len(dataframe), batch_size), desc="Generating embeddings"):
        batch_texts = dataframe[text_column].iloc[i : i + batch_size].tolist()

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

            last_hidden_states = outputs.hidden_states[-1]

            if pooling_method == "mean":
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                sum_embeddings = torch.sum(last_hidden_states * attention_mask, dim=1)
                mean_embeddings = sum_embeddings / attention_mask.sum(dim=1)
                batch_embeddings = mean_embeddings.cpu().numpy()

            elif pooling_method == "max":
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                last_hidden_states[attention_mask == 0] = -float("inf")
                max_embeddings = torch.max(last_hidden_states, dim=1)[0]
                batch_embeddings = max_embeddings.cpu().numpy()

            elif pooling_method == "cls":
                cls_embeddings = last_hidden_states[:, 0, :]
                batch_embeddings = cls_embeddings.cpu().numpy()

            embeddings.append(batch_embeddings)

        if logger is not None and i % print_iter == 0:
            logger.info(f"Generating embeddings {i}/{n}")

    embeddings = np.concatenate(embeddings, axis=0)
    ids = dataframe[index_column].values
    return ids, embeddings


def add_embeddings_faiss_index(
    faiss_index: faiss.Index, ids: np.ndarray, embeddings: np.ndarray
) -> None:
    faiss_index.add_with_ids(embeddings, ids)


def prepare_faiss_df(desc_df_path: str, text_column: str) -> pd.DataFrame:
    desc_df = pd.read_parquet(desc_df_path)
    desc_df[text_column] = desc_df.apply(
        lambda row: " % ".join([row["Title"], row["DescriptionRu"]]), axis=1
    )

    desc_df.drop(columns=["t_rn", "d_rn", "Title", "DescriptionRu"], inplace=True)

    return desc_df
