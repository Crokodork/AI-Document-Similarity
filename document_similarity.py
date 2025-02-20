from sentence_transformers import SentenceTransformer, util

def calculate_similarity(doc1, doc2):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([doc1, doc2], convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return cosine_score.item()

if __name__ == "__main__":
    doc1 = input("Enter the first document text: ")
    doc2 = input("Enter the second document text: ")
    similarity = calculate_similarity(doc1, doc2)
    print(f"Document Similarity Score: {similarity:.2f}")
