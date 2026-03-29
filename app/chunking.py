from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
import time

def fixed_chunking(text):
    """Strategy 1 - Cuts every N characters"""
    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separator="\n"
    )
    return splitter.create_documents([text])


def recursive_chunking(text):
    """Strategy 2 - Tries paragraph → sentence → word → character"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.create_documents([text])


def semantic_chunking(text, embeddings):
    """Strategy 3 - Groups by meaning using embeddings"""
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=70
    )
    return splitter.create_documents([text])


def compare_strategies(text, embeddings):
    """Run all 3 and print comparison table"""
    print("\n" + "="*50)
    print("CHUNKING STRATEGY COMPARISON")
    print("="*50)

    results = {}

    start = time.time()
    fixed = fixed_chunking(text)
    results['Fixed Size'] = {
        'chunks'  : fixed,
        'count'   : len(fixed),
        'avg_size': sum(len(c.page_content) for c in fixed) // len(fixed),
        'time'    : round(time.time() - start, 3)
    }

    start = time.time()
    recursive = recursive_chunking(text)
    results['Recursive'] = {
        'chunks'  : recursive,
        'count'   : len(recursive),
        'avg_size': sum(len(c.page_content) for c in recursive) // len(recursive),
        'time'    : round(time.time() - start, 3)
    }

    start = time.time()
    semantic = semantic_chunking(text, embeddings)
    results['Semantic'] = {
        'chunks'  : semantic,
        'count'   : len(semantic),
        'avg_size': sum(len(c.page_content) for c in semantic) // len(semantic),
        'time'    : round(time.time() - start, 3)
    }

    print(f"\n{'Strategy':<15} {'Chunks':<10} {'Avg Size':<12} {'Time'}")
    print("-" * 50)
    for name, data in results.items():
        print(f"{name:<15} {data['count']:<10} {data['avg_size']:<12} {data['time']}s")
    print("="*50)

    return results