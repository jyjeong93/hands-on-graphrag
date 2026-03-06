from neo4j_graphrag.embeddings.sentence_transformers import SentenceTransformerEmbeddings

embedder = SentenceTransformerEmbeddings(
    model="all-MiniLM-L6-v2",
)

sample_text = "그래프 데이터베이스는 데이터 간의 관계를 저장하는 데 최적화되어 있습니다."
vector = embedder.embed_query(sample_text)

print(f"변환된 벡터의 차원: {len(vector)}") # 384가 출력되어야 함
print(f"벡터 샘플(앞 3개): {vector[:3]}")