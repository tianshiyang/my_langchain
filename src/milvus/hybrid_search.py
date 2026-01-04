#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/1/3 23:45
@Author  : tianshiyang
@File    : hybrid_search.py
"""
import uuid

from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
from milvus import CONNECTION_ARGS
from utils import embeddings

docs = [
    Document(
        page_content="一条测试数据",
        metadata={"category": "Mystery", 'test_data': '这个是测试动态字段'},
    ),
    Document(
        page_content="第二条测试数据",
        metadata={"category": "Mystery", 'test_data_v2': '这个是测试动态字段'},
    ),
    Document(
        page_content="In 'The Last Refuge' by Ethan Blackwood, a group of survivors must band together to escape a post-apocalyptic wasteland, where the last remnants of humanity cling to life in a desperate bid for survival.",
        metadata={"category": "Post-Apocalyptic"},
    ),
    Document(
        page_content="In 'The Memory Thief' by Lila Rose, a charismatic thief with the ability to steal and manipulate memories is hired by a mysterious client to pull off a daring heist, but soon finds themselves trapped in a web of deceit and betrayal.",
        metadata={"category": "Heist/Thriller"},
    ),
    Document(
        page_content="In 'The City of Echoes' by Julian Saint Clair, a brilliant detective must navigate a labyrinthine metropolis where time is currency, and the rich can live forever, but at a terrible cost to the poor.",
        metadata={"category": "Science Fiction"},
    ),
    Document(
        page_content="In 'The Starlight Serenade' by Ruby Flynn, a shy astronomer discovers a mysterious melody emanating from a distant star, which leads her on a journey to uncover the secrets of the universe and her own heart.",
        metadata={"category": "Science Fiction/Romance"},
    ),
    Document(
        page_content="In 'The Shadow Weaver' by Piper Redding, a young orphan discovers she has the ability to weave powerful illusions, but soon finds herself at the center of a deadly game of cat and mouse between rival factions vying for control of the mystical arts.",
        metadata={"category": "Fantasy"},
    ),
    Document(
        page_content="In 'The Lost Expedition' by Caspian Grey, a team of explorers ventures into the heart of the Amazon rainforest in search of a lost city, but soon finds themselves hunted by a ruthless treasure hunter and the treacherous jungle itself.",
        metadata={"category": "Adventure"},
    ),
    Document(
        page_content="In 'The Clockwork Kingdom' by Augusta Wynter, a brilliant inventor discovers a hidden world of clockwork machines and ancient magic, where a rebellion is brewing against the tyrannical ruler of the land.",
        metadata={"category": "Steampunk/Fantasy"},
    ),
    Document(
        page_content="In 'The Phantom Pilgrim' by Rowan Welles, a charismatic smuggler is hired by a mysterious organization to transport a valuable artifact across a war-torn continent, but soon finds themselves pursued by deadly assassins and rival factions.",
        metadata={"category": "Adventure/Thriller"},
    ),
    Document(
        page_content="In 'The Dreamwalker's Journey' by Lyra Snow, a young dreamwalker discovers she has the ability to enter people's dreams, but soon finds herself trapped in a surreal world of nightmares and illusions, where the boundaries between reality and fantasy blur.",
        metadata={"category": "Fantasy"},
    ),
]

COLLECTION_NAME = "hybrid"

def get_milvus_client() -> Milvus:
    dense_index_param = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
    }
    sparse_index_param = {
        "metric_type": "BM25",
        "index_type": "AUTOINDEX",
    }
    vector_store = Milvus(
        embedding_function=embeddings,
        text_field="content",
        primary_field="id",
        vector_field=["vector", "sparse"],
        builtin_function=BM25BuiltInFunction(
            function_name="BM25",
            input_field_names="content",
            output_field_names="sparse"
        ),
        index_params=[dense_index_param, sparse_index_param],
        enable_dynamic_field=True,
        connection_args=CONNECTION_ARGS,
        collection_name=COLLECTION_NAME,
    )
    return vector_store


def insert_to_milvus(vector_store: Milvus):
    print("正在插入数据")
    vector_store.add_documents(
        docs,
        ids=[str(uuid.uuid4()) for _ in range(len(docs))],
    )
    print("插入完成")

def similarity_search_with_ranker(vector_store: Milvus):
    query = "What are the novels Lila has written and what are their contents?"
    result = vector_store.similarity_search(
        query,
        k=1,
        ranker_type="weighted",
        ranker_params={"weights": [0.6, 0.4]}
    )
    print(result)

    # vector_store.similarity_search(query, k=1, ranker_type="rrf", ranker_params={"k": 100})

if __name__ == "__main__":
    milvus = get_milvus_client()
    # insert_to_milvus(milvus)
    similarity_search_with_ranker(milvus)