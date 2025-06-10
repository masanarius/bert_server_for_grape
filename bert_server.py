from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch

# FastAPIのアプリ作成
app = FastAPI()

# ✅ pingエンドポイントを追加（Renderサーバーのスリープ解除に使用）
@app.get("/ping")
def ping():
    return {"message": "pong"}

# Sentence-BERTの日本語モデル（初期ロードに数秒かかります）
model = SentenceTransformer("sonoisa/sentence-bert-base-ja-mean-tokens")

# リクエストのデータ形式
class TextPair(BaseModel):
    text1: str
    text2: str

# 類似度計算エンドポイント
@app.post("/similarity")
def get_similarity(pair: TextPair):
    # 文ベクトルに変換（GPU使用可）
    emb1 = model.encode(pair.text1, convert_to_tensor=True)
    emb2 = model.encode(pair.text2, convert_to_tensor=True)

    # コサイン類似度を計算（スカラ値）
    similarity = util.cos_sim(emb1, emb2).item()

    # 結果を返す
    return {"similarity": similarity}
