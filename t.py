import json

# Dữ liệu advice mẫu từ bạn
advice_sample = {
    'selected_direction': {
        'type': 'Phase-Adaptive Multi-Scale Lookahead with Dynamic Cluster Refinement',
        'rationale': 'By integrating real-time cluster refinement with phase-aware lookahead depth, we achieve 30% better baseline performance while maintaining computational efficiency. The dynamic epsilon adjustment in DBSCAN ensures cluster stability across all problem densities.',
        'what_to_do': [
            'Adaptive epsilon tuning for DBSCAN based on phase-specific density thresholds',
            'Phase-aware lookahead depth scaling with 1-5 step horizon and 3x resolution reduction in critical phases',
            'Hybrid connectivity-density scoring that weights structural bridges during critical phases and density gradients during transitional phases'
        ],
        'expected_impact': 'Projected to reduce the objective value to 5.5-5.9 by creating a unified framework that optimally balances local and global considerations throughout all problem phases'
    },
    'what_to_avoid': [
        {
            'from': 'DeepSeek',
            'critique': "Qwen3-Coder's lazy evaluation pruning fundamentally misunderstands TSP optimization - pruning branches based on local metrics destroys global optimality guarantees and will consistently miss critical bridge connections between clusters. Codestral's phase-adaptive approach is hopelessly over-engineered - DBSCAN clustering with dynamic epsilon has O(n²) worst-case complexity, completely violating their claimed O(n log n) performance, and phase detection heuristics will fail catastrophically on non-uniform problem instances."
        },
        {
            'from': 'Qwen3-Coder',
            'critique': "DeepSeek’s Voronoi-based approach introduces high geometric complexity without bounding its impact on runtime; dynamic partitioning incurs non-trivial recomputation costs under varying densities, risking timeouts. Codestral's multi-phase system assumes predictable structure transitions which break down in real-world instances like clustered-random hybrid graphs, leading to misaligned lookahead scopes and degraded performance."
        }
    ],
    'vote_info': {
        'winner': 'Codestral',
        'votes': 1,
        'total_score': 9,
        'avg_score': 9.0
    },
    'debate_summary': {
        'total_rounds': 3,
        'num_experts': 3
    }
}

# Định nghĩa phương thức (giống như trong class Debate)
def advice_to_narrative(advice):
    if not advice:
        return "No advice generated."

    sel = advice['selected_direction']
    vote = advice['vote_info']
    avoid_list = advice.get('what_to_avoid', [])
    summary = advice['debate_summary']

    narrative = (
        f"After a {summary['total_rounds']}-round debate among {summary['num_experts']} expert LLMs, "
        f"the winning proposal—submitted by **{vote['winner']}** and endorsed by {vote['votes']} out of {summary['num_experts']} experts "
        f"(average score: {vote['avg_score']:.1f}/10)—recommends the following direction:\n\n"
    )

    narrative += f"**{sel['type']}**\n\n"
    narrative += f"{sel['rationale']}\n\n"

    narrative += "Specifically, the approach involves:\n"
    for i, action in enumerate(sel['what_to_do'], 1):
        narrative += f"{i}. {action}\n"
    narrative += f"\nThis is expected to {sel['expected_impact']}.\n"

    if avoid_list:
        narrative += "\nThe debate also highlighted critical pitfalls to avoid:\n"
        for item in avoid_list:
            critique_text = item['critique'].strip()
            # Loại bỏ dấu ngoặc kép thừa ở đầu/cuối (nếu có)
            if critique_text.startswith('"') and critique_text.endswith('"'):
                critique_text = critique_text[1:-1]
            narrative += f"- **{item['from']}** warned: \"{critique_text}\"\n"

    return narrative.strip()

# Chạy thử
if __name__ == "__main__":
    output = advice_to_narrative(advice_sample)
    print(output)