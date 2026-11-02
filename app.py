import os
from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px

CACHE_FILE = "embeddings_cache.pkl"
TSNE_CACHE_FILE = "tsne_3d_cache.npy"
BATCH_SIZE = 64
MODEL = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL)


def build_dataframe():
    ds = load_dataset("myyycroft/RiDiC_cars_self_consistency_15_generations")
    df = ds["train"].to_pandas()
    df = df[["title_en", "country", "gpt-5-chat"]].copy()

    texts = df['gpt-5-chat'].fillna("").tolist()
    all_embeddings = [None] * len(texts)

    valid_indices = [i for i, t in enumerate(texts) if t.strip() != ""]
    valid_texts = [texts[i] for i in valid_indices]

    embeddings = model.encode(
        valid_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True
    )

    # Розкладаємо ембединги по правильних позиціях
    for idx, emb in zip(valid_indices, embeddings):
        all_embeddings[idx] = emb.tolist()

    df['embedding'] = all_embeddings
    return df


if os.path.exists(CACHE_FILE):
    df = pd.read_pickle(CACHE_FILE)
    print(f"Завантажено з кешу: {CACHE_FILE}")
else:
    df = build_dataframe()
    df.to_pickle(CACHE_FILE)
    print(f"Збережено в кеш: {CACHE_FILE}")

success_count = df['embedding'].notna().sum()
print(f"\n✅ Ембединги готові для {success_count}/{len(df)} записів!")

print(df.head())

def cosine_similarity(a, b):
    """Обчислює cosine similarity між двома векторами"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_similar_cars(query_text, df, top_n=5):
    """Знаходить ТОП-N схожих авто за описом"""

    # Створюємо ембединг для запиту тією ж моделлю, що й для датасету
    query_embedding = model.encode(query_text)

    # Фільтруємо записи з ембедингами
    df_with_emb = df[df['embedding'].notna()].copy()

    # Обчислюємо cosine similarity для кожного запису
    df_with_emb['similarity'] = df_with_emb['embedding'].apply(
        lambda emb: cosine_similarity(query_embedding, emb)
    )

    # Сортуємо та повертаємо топ-N
    return df_with_emb.nlargest(top_n, 'similarity')[
        ['title_en', 'country', 'gpt-5-chat', 'similarity']
    ]

query = "electric car with 450 horse power from United States"

print(f"🔍 Шукаємо авто схожі на: \"{query}\"\n")
print("=" * 80)

results = find_similar_cars(query, df)

for i, (_, row) in enumerate(results.iterrows(), 1):
    print(f"\n🎬 #{i}: {row['title_en']}")
    print(f"⭐ Країна: {row['country']}")
    print(f"📊 Схожість: {row['similarity']:.4f}")
    print(f"📝 {row['gpt-5-chat'][:200]}...")
    print("-" * 80)


# ---------- 3D візуалізація ембедингів (t-SNE) ----------

df_vis = df[df['embedding'].notna()].copy()
X = np.array(df_vis['embedding'].tolist())

if os.path.exists(TSNE_CACHE_FILE) and len(np.load(TSNE_CACHE_FILE)) == len(X):
    coords = np.load(TSNE_CACHE_FILE)
    print(f"t-SNE завантажено з кешу: {TSNE_CACHE_FILE}")
else:
    print("Обчислюю t-SNE (може зайняти кілька хвилин)...")
    tsne = TSNE(
        n_components=3,
        perplexity=30,
        metric="cosine",
        init="random",
        random_state=42,
    )
    coords = tsne.fit_transform(X)
    np.save(TSNE_CACHE_FILE, coords)
    print(f"t-SNE збережено в кеш: {TSNE_CACHE_FILE}")

df_vis[['x', 'y', 'z']] = coords

fig = px.scatter_3d(
    df_vis,
    x='x', y='y', z='z',
    color='country',
    hover_name='title_en',
    opacity=0.8,
)
fig.update_traces(marker_size=3)
fig.show()
