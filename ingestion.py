!pip install -q groq chromadb
import os
import chromadb
from chromadb.utils import embedding_functions

data_path = "./university_data"
collection_name = "gtu_docs"

client = chromadb.PersistentClient(path=data_path)
ef = embedding_functions.DefaultEmbeddingFunction()
collection = client.get_or_create_collection(name=collection_name, embedding_function=ef)

documents = []
metadatas = []
ids = []

print(f"Indexing files from {data_path}...")

file_list = [f for f in os.listdir(data_path) if f.endswith('.txt')]

for filename in file_list:
    with open(os.path.join(data_path, filename), 'r') as f:
        content = f.read()

        chunks = [c.strip() for c in content.split('\n\n') if len(c.strip()) > 20]
        for j, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"source": filename})
            ids.append(f"{filename}_{j}")


batch_size = 100
for i in range(0, len(documents), batch_size):
    collection.add(
        documents=documents[i:i+batch_size],
        metadatas=metadatas[i:i+batch_size],
        ids=ids[i:i+batch_size]
    )

print(f"✅ SUCCESS: {collection.count()} chunks indexed into '{collection_name}'.")
print("You can now run your main Groq Agent code cell!")
