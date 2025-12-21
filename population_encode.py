from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, euclidean
from scipy.sparse.csgraph import minimum_spanning_tree
from math import log
import os



# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Define a function to get embeddings


def compute_cosine_similarity(embeddings):
    return cosine_similarity(embeddings)


def cluster_nodes(similarity_matrix, threshold=0.95):
    # Convert similarity matrix to distance matrix
    distance_matrix = 1 - similarity_matrix
    distance_matrix = np.clip(distance_matrix, 0, None)

    # Use hierarchical clustering
    condensed_distance_matrix = squareform(distance_matrix)
    Z = linkage(condensed_distance_matrix, method='complete')

    clusters = fcluster(Z, t=1 - threshold, criterion='distance')

    # Collect nodes into clusters
    node_clusters = {}
    for node, cluster_id in enumerate(clusters):
        if cluster_id not in node_clusters:
            node_clusters[cluster_id] = []
        node_clusters[cluster_id].append(node)

    return list(node_clusters.values())


# Model and tokenizer setup
checkpoint = "Salesforce/codet5p-110m-embedding"
device = "cpu"  # Use "cuda" if you have a GPU
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

import os
import autopep8
import tempfile
import json
import re

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)


def format_python_code(code_str):
    # Write the code string to a temporary file
    with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8"
    ) as temp_file:
        temp_file.write(code_str)
        temp_filename = temp_file.name

    # Format the file using autopep8
    autopep8_args = ['--in-place', '--aggressive', '--aggressive', temp_filename]
    autopep8.fix_file(temp_filename, options=autopep8.parse_args(autopep8_args))

    # Read the formatted code from the file
    with open(temp_filename, 'r') as temp_file:
        formatted_code = temp_file.read()

    return formatted_code


def remove_comments_and_docstrings(code):
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node

    lines = code.split('\n')
    to_remove = []

    def is_comment(node):
        return node.type == 'comment'

    def is_docstring(node):
        return (node.type == 'string' and
                (node.parent.type in ['expression_statement', 'module']))

    def traverse(node):
        if is_comment(node):
            to_remove.append((node.start_point, node.end_point))
        elif is_docstring(node):
            to_remove.append((node.start_point, node.end_point))
        for child in node.children:
            traverse(child)

    traverse(root_node)

    for start, end in reversed(to_remove):
        start_row, start_col = start
        end_row, end_col = end
        if start_row == end_row:
            lines[start_row] = lines[start_row][:start_col] + lines[start_row][end_col:]
        else:
            lines[start_row] = lines[start_row][:start_col]
            for row in range(start_row + 1, end_row):
                lines[row] = ''
            lines[end_row] = lines[end_row][end_col:]

    cleaned_lines = [line for line in lines if line.strip() != '']

    return '\n'.join(cleaned_lines)


code = '''
def select_next_node_v2(current_node, destination_node, unvisited_nodes, distance_matrix):
    """{This algorithm selects the nearest unvisited node to the current node, but prioritizes nodes that are closer to the destination node by a weighted factor.}"""

    best_node = None
    min_cost = float('inf')

    for neighbor in unvisited_nodes:
        cost = distance_matrix[current_node][neighbor] + 0.5 * distance_matrix[neighbor][destination_node]
        if cost < min_cost:
            min_cost = cost
            best_node = neighbor

    return next_node
'''

code_2 = '''
def select_next_node_v3(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    This heuristic chooses the next node among unvisited ones by minimizing
    a weighted cost: distance to neighbor + 0.5 * distance from neighbor to destination.
    """
    candidate_nodes = {}
    for node in unvisited_nodes:
        cost = distance_matrix[current_node][node] + 0.5 * distance_matrix[node][destination_node]
        candidate_nodes[node] = cost

    # Pick the node with minimum cost
    next_node = min(candidate_nodes, key=candidate_nodes.get)
    return next_node
'''


def get_embedding(code, model=model, tokenizer=tokenizer, device='cpu'):
    inputs = tokenizer.encode(code, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model(inputs)[0].cpu().numpy()
    return np.array(embedding.reshape(1, -1)).squeeze()


def similarity(code_1, code_2):
    v1 = get_embedding(remove_comments_and_docstrings(format_python_code(code_1)))
    v2 = get_embedding(remove_comments_and_docstrings(format_python_code(code_2)))

    # Stack embeddings into a 2D array for pairwise similarity
    embeddings = [v1, v2]

    # Compute similarity matrix
    sim_matrix = compute_cosine_similarity(embeddings)

    return sim_matrix[0, 1]


def cluster_nodes(similarity_matrix, threshold=0.95):
    """
    Cluster nodes based on similarity matrix using hierarchical clustering.

    Args:
        similarity_matrix: Square matrix of pairwise similarities (values from 0 to 1)
        threshold: Similarity threshold for clustering (default 0.95)

    Returns:
        List of clusters, where each cluster is a list of node indices
    """
    # Convert similarity matrix to distance matrix
    distance_matrix = 1 - similarity_matrix
    distance_matrix = np.clip(distance_matrix, 0, None)

    # Handle edge case: single node
    if len(distance_matrix) == 1:
        return [[0]]

    # Use hierarchical clustering
    try:
        condensed_distance_matrix = squareform(distance_matrix, checks=False)
        Z = linkage(condensed_distance_matrix, method='complete')
        clusters = fcluster(Z, t=1 - threshold, criterion='distance')
    except ValueError as e:
        print(f"Clustering error: {e}")
        # Fallback: each node is its own cluster
        return [[i] for i in range(len(distance_matrix))]

    # Collect nodes into clusters
    node_clusters = {}
    for node, cluster_id in enumerate(clusters):
        if cluster_id not in node_clusters:
            node_clusters[cluster_id] = []
        node_clusters[cluster_id].append(node)

    return list(node_clusters.values())


def calculate_shannon_diversity(clusters, total_nodes):
    """
    Calculate Shannon Diversity Index for clusters.

    Args:
        clusters: List of clusters
        total_nodes: Total number of nodes

    Returns:
        Shannon diversity index value
    """
    if total_nodes == 0:
        return 0.0

    proportions = [len(cluster) / total_nodes for cluster in clusters]
    shannon_index = -sum(p * log(p) for p in proportions if p > 0)
    return shannon_index


def shannon_diversity(embeddings, threshold=0.95):
    """
    Compute Shannon Diversity Index based on code embeddings.

    Args:
        embeddings: List or array of embedding vectors
        threshold: Similarity threshold for clustering

    Returns:
        Shannon diversity index value
    """
    # Ensure embeddings is a proper numpy array
    if isinstance(embeddings, list):
        embeddings = [np.array(e).flatten() for e in embeddings]

    embeddings_2d = np.vstack(embeddings)

    # Handle single embedding case
    if len(embeddings_2d) == 1:
        print("Clusters: [[0]]")
        print("Shannon Diversity Index: 0.0")
        return 0.0

    # Compute similarity matrix
    similarity_matrix = compute_cosine_similarity(embeddings_2d)
    np.fill_diagonal(similarity_matrix, 1)

    # Cluster nodes with a similarity threshold
    clusters = cluster_nodes(similarity_matrix, threshold)

    # Calculate Shannon Diversity Index
    total_nodes = len(embeddings)
    shannon_diversity_index = calculate_shannon_diversity(clusters, total_nodes)

    print("Clusters:", clusters)
    print("Shannon Diversity Index:", shannon_diversity_index)
    return shannon_diversity_index


def total_diversity(embeddings):
    """
    Calculate diversity index using Minimum Spanning Tree approach.

    Args:
        embeddings: List or array of embedding vectors

    Returns:
        Code Diversity Index (CDI) value
    """
    # Ensure embeddings is properly formatted
    if isinstance(embeddings, list):
        embeddings = [np.array(e).flatten() for e in embeddings]

    embeddings_1d = np.vstack(embeddings)

    # Handle single embedding case
    if len(embeddings_1d) == 1:
        print("CDI: 0.0")
        return 0.0

    # Compute the distance matrix more efficiently
    n = len(embeddings_1d)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            distance = euclidean(embeddings_1d[i], embeddings_1d[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    # Find the Minimum Spanning Tree (MST)
    mst = minimum_spanning_tree(distance_matrix).toarray()

    # Calculate the diversity index
    mst_distances = mst[mst > 0]  # Changed != to > to handle floating point

    if len(mst_distances) == 0:
        print("CDI: 0.0")
        return 0.0

    total_distance = np.sum(mst_distances)

    if total_distance == 0:
        print("CDI: 0.0")
        return 0.0

    proportions = mst_distances / total_distance
    diversity_index = -np.sum(proportions * np.log(proportions))

    print(f"CDI: {diversity_index}")
    return diversity_index


import json


def read_algorithms_from_file(file_path):
    """
    Đọc file JSON chứa danh sách các thuật toán.

    Args:
        file_path: Đường dẫn đến file JSON

    Returns:
        List các dictionary chứa thông tin thuật toán
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            algorithms = json.load(f)
        return algorithms
    except FileNotFoundError:
        print(f"Không tìm thấy file: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Lỗi khi đọc JSON: {e}")
        return []


def extract_codes_from_algorithms(algorithms):
    """
    Trích xuất tất cả đoạn code từ danh sách thuật toán.

    Args:
        algorithms: List các dictionary chứa thông tin thuật toán

    Returns:
        List các đoạn code (string)
    """
    codes = []
    for algo in algorithms:
        if 'code' in algo and algo['code']:
            codes.append(algo['code'])
    return codes


def get_embeddings_for_algorithms(algorithms):
    """
    Tạo embeddings cho tất cả các thuật toán trong file.

    Args:
        algorithms: List các dictionary chứa thông tin thuật toán

    Returns:
        List các embedding vectors
    """
    codes = extract_codes_from_algorithms(algorithms)
    embeddings = []

    for i, code in enumerate(codes):
        try:
            # Xử lý và tạo embedding
            processed_code = remove_comments_and_docstrings(format_python_code(code))
            embedding = get_embedding(processed_code)
            embeddings.append(embedding)
            print(f"Đã tạo embedding cho thuật toán {i + 1}/{len(codes)}")
        except Exception as e:
            print(f"Lỗi khi xử lý thuật toán {i + 1}: {e}")
            continue

    return embeddings


def analyze_algorithm_diversity(file_path, threshold=0.95):
    """
    Phân tích độ đa dạng của các thuật toán trong file.

    Args:
        file_path: Đường dẫn đến file JSON
        threshold: Ngưỡng similarity cho clustering

    Returns:
        Dictionary chứa kết quả phân tích
    """
    # Đọc file
    algorithms = read_algorithms_from_file(file_path)

    if not algorithms:
        print("Không có dữ liệu để phân tích")
        return None

    print(f"Đã đọc {len(algorithms)} thuật toán từ file")

    # Tạo embeddings
    print("\nĐang tạo embeddings...")
    embeddings = get_embeddings_for_algorithms(algorithms)

    if len(embeddings) == 0:
        print("Không thể tạo embedding nào")
        return None

    print(f"\nĐã tạo {len(embeddings)} embeddings")

    # Tính toán diversity
    print("\n" + "=" * 50)
    print("PHÂN TÍCH ĐỘ ĐA DẠNG")
    print("=" * 50)

    # Shannon Diversity Index
    print("\n--- Shannon Diversity Index ---")
    shannon_div = shannon_diversity(embeddings, threshold=threshold)

    # Total Diversity Index
    print("\n--- Code Diversity Index (CDI) ---")
    total_div = total_diversity(embeddings)

    # Tính similarity matrix
    print("\n--- Similarity Matrix ---")
    embeddings_2d = np.vstack(embeddings)
    similarity_matrix = compute_cosine_similarity(embeddings_2d)

    # Thống kê similarity
    n = len(embeddings)
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            similarities.append(similarity_matrix[i, j])

    print(f"Similarity trung bình: {np.mean(similarities):.4f}")
    print(f"Similarity cao nhất: {np.max(similarities):.4f}")
    print(f"Similarity thấp nhất: {np.min(similarities):.4f}")
    print(f"Độ lệch chuẩn: {np.std(similarities):.4f}")

    # Tìm các cặp thuật toán tương đồng nhất
    print("\n--- Top 5 cặp thuật toán giống nhau nhất ---")
    similarity_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            similarity_pairs.append((i, j, similarity_matrix[i, j]))

    similarity_pairs.sort(key=lambda x: x[2], reverse=True)

    for idx, (i, j, sim) in enumerate(similarity_pairs[:5], 1):
        print(f"{idx}. Thuật toán {i + 1} vs Thuật toán {j + 1}: {sim:.4f}")
        print(f"   Objective {i + 1}: {algorithms[i].get('objective', 'N/A')}")
        print(f"   Objective {j + 1}: {algorithms[j].get('objective', 'N/A')}")

    results = {
        'num_algorithms': len(algorithms),
        'num_embeddings': len(embeddings),
        'shannon_diversity': shannon_div,
        'code_diversity': total_div,
        'avg_similarity': np.mean(similarities),
        'max_similarity': np.max(similarities),
        'min_similarity': np.min(similarities),
        'std_similarity': np.std(similarities),
        'similarity_matrix': similarity_matrix,
        'algorithms': algorithms
    }

    return results


# Sử dụng
if __name__ == "__main__":
    file_path = r"D:\AdaptiveMCTSAHD\outputs\tsp_constructive-constructive\ab-mcts-ahd\2025-12-11_22-44-44\population_generation_82.json"  # Thay bằng đường dẫn file của bạn

    # Phân tích với threshold mặc định
    results = analyze_algorithm_diversity(file_path, threshold=0.95)

    # Nếu muốn thử các threshold khác
    print("\n" + "=" * 50)
    print("PHÂN TÍCH VỚI CÁC THRESHOLD KHÁC NHAU")
    print("=" * 50)

    algorithms = read_algorithms_from_file(file_path)
    embeddings = get_embeddings_for_algorithms(algorithms)

    for threshold in [0.90, 0.95, 0.98]:
        print(f"\nThreshold = {threshold}")
        shannon_diversity(embeddings, threshold=threshold)



