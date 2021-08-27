from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

print("Max Sequence Length:", model.max_seq_length)

# Change the length to 200
model.max_seq_length = 200

print("Max Sequence Length:", model.max_seq_length)
tensors