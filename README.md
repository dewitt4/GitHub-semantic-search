# GitHub-semantic-search
GitHub Semantic Search Optimizer

# GitHub Semantic Search Indexer

A powerful tool that enables semantic search capabilities for GitHub repositories. Instead of just searching for exact matches, this tool understands the meaning behind your queries and finds relevant code and documentation across your repository.

## Features

- 🔍 Semantic search using OpenAI's text-embedding-3-small model
- 💾 Efficient vector storage with ChromaDB
- 🔄 Automatic repository cloning and updating
- 📄 Support for multiple file types including Python, JavaScript, Java, and more
- 📦 Intelligent text chunking for better search results
- 🚀 Batch processing for better performance
- 📊 Progress tracking for long-running operations

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/github-semantic-search
cd github-semantic-search
```

2. Install the required packages:
```bash
pip install chromadb openai tiktoken gitpython tqdm
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key'
```

## Usage

### Basic Usage

```python
from github_search import GitHubSearchIndexer

# Initialize the indexer
indexer = GitHubSearchIndexer(
    repo_url="https://github.com/username/repository",
    target_dir="repo_data"
)

# Index the repository
indexer.index_repository()

# Search the repository
results = indexer.search("database connection handling")

# Print results
for result in results:
    print(f"\nFile: {result['path']}")
    print(f"Relevance Score: {1 - result['distance']:.4f}")
    print("Content snippet:")
    print(result['content'][:200] + "...")
```

### Customizing Search

You can customize the number of results:

```python
# Get top 10 results
results = indexer.search("database connection handling", n_results=10)
```

### Supported File Types

The indexer processes the following file extensions:
- Python (.py)
- JavaScript (.js, .jsx)
- TypeScript (.tsx)
- Java (.java)
- C++ (.cpp, .h)
- C# (.cs)
- Ruby (.rb)
- PHP (.php)
- Go (.go)
- Rust (.rs)
- Documentation (.md, .rst, .txt)
- Configuration files (.json, .yml, .yaml, .toml, .ini)

## How It Works

1. **Repository Cloning**: The tool clones the target repository or updates it if it already exists.

2. **File Processing**: 
   - Files are read and processed based on their file extensions
   - Large files are automatically split into smaller chunks for better search accuracy
   - Text is encoded using the cl100k_base tokenizer

3. **Embedding Generation**: 
   - Each chunk of text is converted into embeddings using OpenAI's text-embedding-3-small model
   - Embeddings capture the semantic meaning of the code and documentation

4. **Vector Storage**: 
   - Embeddings are stored efficiently using ChromaDB
   - Enables fast similarity search and retrieval

5. **Search**: 
   - Queries are converted to embeddings
   - ChromaDB finds the most similar chunks based on embedding similarity
   - Results are returned with relevance scores and file metadata

## Configuration

The indexer accepts several parameters:

```python
GitHubSearchIndexer(
    repo_url: str,          # URL of the GitHub repository
    target_dir: str = "repo_data",  # Directory to store repository and index
)
```

Additional configuration during indexing:

```python
indexer.index_repository(
    batch_size: int = 100   # Number of documents to process in each batch
)
```

## Performance Considerations

- The initial indexing process may take some time depending on:
  - Repository size
  - Number of files
  - OpenAI API response time
- Subsequent updates are faster as they only process changed files
- Search queries are typically very fast due to ChromaDB's efficient similarity search

## Dependencies

- `chromadb`: Vector storage and similarity search
- `openai`: Text embedding generation
- `tiktoken`: Text tokenization
- `gitpython`: Repository management
- `tqdm`: Progress tracking

## Error Handling

The tool includes robust error handling:
- Gracefully handles different file encodings
- Skips problematic files with error reporting
- Validates API key availability
- Provides clear error messages for common issues

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Uses OpenAI's text-embedding-3-small model for high-quality embeddings
- Built with ChromaDB for efficient vector storage
- Inspired by the need for better code search tools
