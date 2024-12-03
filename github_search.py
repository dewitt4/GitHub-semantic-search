import os
from git import Repo
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict, Tuple
import json

class GitHubSemanticSearch:
    def __init__(self, repo_url: str, target_dir: str = "repo_data"):
        """
        Initialize the semantic search indexer.
        
        Args:
            repo_url (str): URL of the GitHub repository
            target_dir (str): Directory to store the repository and index
        """
        self.repo_url = repo_url
        self.target_dir = Path(target_dir)
        self.repo_path = self.target_dir / "repo"
        self.index_path = self.target_dir / "index"
        
        # Initialize the transformer model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        # Create directories if they don't exist
        self.target_dir.mkdir(exist_ok=True)
        self.index_path.mkdir(exist_ok=True)

    def clone_repository(self):
        """Clone or update the repository."""
        if self.repo_path.exists():
            repo = Repo(self.repo_path)
            origin = repo.remotes.origin
            origin.pull()
        else:
            Repo.clone_from(self.repo_url, self.repo_path)

    def get_file_content(self, file_path: Path) -> str:
        """Read and return file content, handling different encodings."""
        try:
            return file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            return file_path.read_text(encoding='latin-1')

    def create_embeddings(self, text: str) -> np.ndarray:
        """Create embeddings for the given text using the transformer model."""
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()[0]

    def process_files(self) -> List[Dict]:
        """Process all files in the repository and create embeddings."""
        documents = []
        
        # File extensions to process
        valid_extensions = {'.py', '.js', '.java', '.cpp', '.h', '.cs', '.rb', '.php', '.txt', '.md', '.rst'}
        
        for file_path in self.repo_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in valid_extensions:
                try:
                    relative_path = file_path.relative_to(self.repo_path)
                    content = self.get_file_content(file_path)
                    
                    # Create chunks of the content
                    chunks = self.create_chunks(content)
                    
                    for i, chunk in enumerate(chunks):
                        embedding = self.create_embeddings(chunk)
                        
                        documents.append({
                            'path': str(relative_path),
                            'chunk_id': i,
                            'content': chunk,
                            'embedding': embedding.tolist()
                        })
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
        
        return documents

    def create_chunks(self, content: str, chunk_size: int = 1000) -> List[str]:
        """Split content into chunks of approximately equal size."""
        words = content.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def save_index(self, documents: List[Dict]):
        """Save the processed documents and embeddings."""
        index_file = self.index_path / 'index.json'
        with open(index_file, 'w') as f:
            json.dump(documents, f)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Search the indexed repository using semantic similarity.
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            
        Returns:
            List of tuples containing (file_path, content_chunk, similarity_score)
        """
        # Load the index
        with open(self.index_path / 'index.json', 'r') as f:
            documents = json.load(f)
        
        # Create query embedding
        query_embedding = self.create_embeddings(query)
        
        # Calculate similarities
        similarities = []
        for doc in documents:
            embedding = np.array(doc['embedding'])
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((doc['path'], doc['content'], similarity))
        
        # Sort by similarity and return top_k results
        return sorted(similarities, key=lambda x: x[2], reverse=True)[:top_k]

    def index_repository(self):
        """Main method to index the repository."""
        print("Cloning/updating repository...")
        self.clone_repository()
        
        print("Processing files and creating embeddings...")
        documents = self.process_files()
        
        print("Saving index...")
        self.save_index(documents)
        print("Indexing complete!")

# Example usage
if __name__ == "__main__":
    searcher = GitHubSemanticSearch(
        repo_url="https://github.com/username/repository",
        target_dir="indexed_repo"
    )
    
    # Index the repository
    searcher.index_repository()
    
    # Perform a search
    results = searcher.search("database connection handling")
    
    # Print results
    for path, content, score in results:
        print(f"\nFile: {path}")
        print(f"Score: {score:.4f}")
        print("Content snippet:")
        print(content[:200] + "...")
