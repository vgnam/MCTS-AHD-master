import re
import json
from .interface_LLM import InterfaceAPI as InterfaceLLM


class Debate():
    """
    Class để quản lý debate giữa nhiều LLM và trả về advice/direction
    """

    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode, prompts, **kwargs):
        """
        Args:
            api_endpoint: API endpoint cho LLM chính (không dùng trong debate)
            api_key: API key cho LLM chính (không dùng trong debate)
            model_LLM: Model name cho LLM chính (không dùng trong debate)
            debug_mode: bool
            prompts: Prompts object
            **kwargs: Phải chứa:
                - use_local_llm: bool
                - url: str
                - advisor_configs: List of dict cho các LLM cố vấn
                    Mỗi config có thể chứa:
                    - 'api_endpoint', 'api_key', 'model': API configs
                    - 'name': Tên LLM (e.g., 'GPT4', 'Claude')
                    - 'role': Vai trò trong debate (e.g., 'Optimization Expert', 'Algorithm Designer')
                    - 'expertise': Lĩnh vực chuyên môn (e.g., 'combinatorial optimization', 'machine learning')
                - debate_rounds: int (default=3)
        """
        assert 'use_local_llm' in kwargs
        assert 'url' in kwargs
        assert 'advisor_configs' in kwargs, "Must provide advisor_configs for debate"

        self._use_local_llm = kwargs.get('use_local_llm')
        self._url = kwargs.get('url')

        # Set prompts
        self.prompt_task = prompts.get_task()
        self.prompt_func_name = prompts.get_func_name()
        self.prompt_func_inputs = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf = prompts.get_inout_inf()
        self.prompt_other_inf = prompts.get_other_inf()

        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"

        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"

        # Settings
        self.debug_mode = debug_mode
        self.debate_rounds = kwargs.get('debate_rounds', 3)

        # Khởi tạo các LLM cố vấn với vai trò
        advisor_configs = kwargs.get('advisor_configs', [])
        self.advisor_llms = []

        # Default roles nếu không được chỉ định
        default_roles = [
            "Optimization Expert",
            "Algorithm Designer",
            "Performance Analyst",
            "Innovation Specialist",
            "Critical Reviewer"
        ]

        default_expertise = [
            "combinatorial optimization and mathematical programming",
            "algorithm design and data structures",
            "computational complexity and efficiency analysis",
            "novel heuristics and meta-heuristics",
            "robustness and edge case analysis"
        ]

        for i, config in enumerate(advisor_configs):
            llm = InterfaceLLM(
                config['api_endpoint'],
                config['api_key'],
                config['model'],
                debug_mode
            )
            self.advisor_llms.append({
                'interface': llm,
                'name': config.get('name', f"Expert_{i + 1}"),
                'role': config.get('role', default_roles[i % len(default_roles)]),
                'expertise': config.get('expertise', default_expertise[i % len(default_expertise)]),
                'performance_history': []
            })

        if self.debug_mode:
            print(f"Debate initialized with {len(self.advisor_llms)} experts:")
            for llm_info in self.advisor_llms:
                print(f"  - {llm_info['name']}: {llm_info['role']} (expertise: {llm_info['expertise']})")

    def get_prompt_initial_direction(self, heuristic_info, llm_info):
        """
        Prompt cho vòng đầu tiên: đề xuất direction với vai trò cụ thể
        """
        prompt = f"""You are {llm_info['name']}, a {llm_info['role']} participating in a competitive expert debate to improve this heuristic algorithm.

COMPETITIVE CHALLENGE:
You are competing with other experts to propose the BEST improvement direction. Your goal is to identify weaknesses in the current approach and propose a direction that will outperform both the current baseline and other experts' suggestions.

TASK: {self.prompt_task}
Function signature: {self.prompt_func_name}({self.joined_inputs}) -> {self.joined_outputs}
{self.prompt_inout_inf}
{self.prompt_other_inf}

CURRENT BASELINE TO BEAT:
Algorithm Description: {heuristic_info.get('algorithm', 'N/A')}
Objective Value: {heuristic_info.get('objective', 'N/A')} (LOWER IS BETTER)

Current Implementation:
{heuristic_info.get('code', 'N/A')}

DEBATE STRUCTURE:
- Round 0 (NOW): Each expert analyzes and proposes one improvement direction
- Rounds 1-{self.debate_rounds}: Competitive debate where you critique opponents' ideas, defend yours, and refine your strategy
- Final Round: All experts vote to select the winning proposal

YOUR COMPETITIVE STRATEGY:
1. First, ANALYZE the current baseline thoroughly:
   - What weaknesses can you exploit?
   - Where does it fail or perform suboptimally?
   - What opportunities for improvement exist?

2. Then, PROPOSE one improvement direction that will outperform the baseline

REQUIREMENTS:
- Be specific and actionable
- Focus on measurable improvements
- Consider implementation feasibility
- Aim to beat the current objective value

Respond in JSON format:
{{
    "reasoning": "Deep analysis of current baseline: strengths, critical weaknesses, failure modes, and improvement opportunities (3-5 sentences)",
    "direction_type": "Type of improvement (e.g., 'Add adaptive mechanism', 'Restructure priority logic', 'Introduce new term')",
    "rationale": "Why this direction will beat the baseline objective (2-3 sentences with specific mechanisms)",
    "key_aspects": ["Specific implementation point 1", "Specific implementation point 2", "Specific implementation point 3"],
    "expected_impact": "Quantitative or qualitative improvement expected (be specific)"
}}
"""
        return prompt

    def get_prompt_debate_round(self, heuristic_info, llm_info, own_direction, other_directions, round_num):
        """Prompt cho các vòng debate tiếp theo"""

        other_experts_text = ""
        for direction in other_directions:
            other_experts_text += f"\n{'=' * 60}\n"
            other_experts_text += f"{direction['llm_name']}'s PROPOSAL:\n"
            other_experts_text += f"  Analysis: {direction.get('reasoning', 'N/A')[:150]}...\n"
            other_experts_text += f"  Direction: {direction['direction_type']}\n"
            other_experts_text += f"  Why it wins: {direction['rationale']}\n"
            other_experts_text += f"  Key aspects: {', '.join(direction['key_aspects'])}\n"

        prompt = f"""You are {llm_info['name']} in ROUND {round_num}/{self.debate_rounds} of the competitive debate.

BASELINE OBJECTIVE: {heuristic_info.get('objective', 'N/A')} (LOWER IS BETTER)

YOUR CURRENT PROPOSAL:
Analysis: {own_direction.get('reasoning', 'N/A')}
Direction: {own_direction['direction_type']}
Rationale: {own_direction['rationale']}
Key aspects: {', '.join(own_direction['key_aspects'])}
Expected impact: {own_direction.get('expected_impact', 'N/A')}

COMPETING PROPOSALS:
{other_experts_text}

COMPETITIVE DEBATE OBJECTIVES:
This is a competitive optimization challenge. You must:

1. ATTACK opponents' proposals:
   - Identify fatal flaws or weak points in their logic
   - Point out why their approaches won't beat the baseline
   - Highlight overlooked edge cases or failure modes

2. DEFEND your proposal:
   - Counter any potential criticisms
   - Demonstrate why your approach is superior
   - Provide evidence or reasoning for effectiveness

3. REFINE your strategy:
   - Incorporate insights that strengthen your proposal
   - Address any weaknesses opponents might exploit
   - Enhance your competitive advantage

You can completely change your proposal if you discover a superior strategy.

CRITICAL RULES:
- Focus on beating the baseline objective: {heuristic_info.get('objective', 'N/A')}
- Be aggressive in identifying flaws in competing proposals
- Be specific about implementation details
- Aim for maximum competitive advantage

Respond in JSON format:
{{
    "reasoning": "Your strategic analysis of the competitive landscape in this round (2-3 sentences)",
    "critique_others": "Aggressive critique of opponents' proposals - identify their critical weaknesses and why they will fail (be specific)",
    "defense": "Strong defense of your proposal - why it will outperform both baseline and competitors (be convincing)",
    "refined_direction": {{
        "direction_type": "Your refined/new direction",
        "rationale": "Enhanced rationale with competitive advantage (2-3 sentences)",
        "key_aspects": ["Refined aspect 1", "Refined aspect 2", "Refined aspect 3"],
        "expected_impact": "Updated expected improvement",
        "changes_from_previous": "What you changed and why it strengthens your competitive position"
    }}
}}
"""
        return prompt

    def get_prompt_vote(self, all_directions, llm_info, heuristic_info):
        """Prompt để vote cho direction tốt nhất"""

        proposals_text = ""
        for direction in all_directions:
            if direction['llm_name'] != llm_info['name']:  # Không vote cho chính mình
                proposals_text += f"\n{'=' * 60}\n"
                proposals_text += f"{direction['llm_name']}'s FINAL PROPOSAL:\n"
                proposals_text += f"  Direction: {direction['direction_type']}\n"
                proposals_text += f"  Rationale: {direction['rationale']}\n"
                proposals_text += f"  Key aspects: {', '.join(direction['key_aspects'])}\n"
                proposals_text += f"  Expected impact: {direction.get('expected_impact', 'N/A')}\n"

        prompt = f"""You are {llm_info['name']}. After {self.debate_rounds} rounds of competitive debate, you must now vote for the WINNING proposal.

BASELINE TO BEAT: Objective = {heuristic_info.get('objective', 'N/A')} (LOWER IS BETTER)

COMPETING PROPOSALS (excluding yours):
{proposals_text}

VOTING OBJECTIVE:
Select the proposal most likely to BEAT THE BASELINE and outperform all other proposals.

EVALUATION CRITERIA:
1. **Impact Potential**: Will this actually beat the baseline objective?
2. **Implementation Feasibility**: Can it be coded correctly and efficiently?
3. **Robustness**: Will it work across different cases and edge cases?
4. **Competitive Advantage**: What unique strength makes this superior?

YOUR VOTE:
First, reason about which proposal has the highest probability of winning. Then cast your vote.

REQUIREMENTS:
- Be objective and analytical
- Focus on competitive performance
- Justify your decision with specific technical reasoning

Respond in JSON format:
{{
    "reasoning": "Your competitive analysis: which proposal will most likely beat the baseline and why (3-4 sentences with specific technical reasons)",
    "best_direction_llm": "Name of expert with winning proposal",
    "why_best": "Specific competitive advantages that make this proposal superior (2-3 concrete reasons)",
    "score": 8,
    "weaknesses_to_address": "Any remaining weaknesses in the winning proposal that should be handled during implementation"
}}

Score scale (1-10):
- 9-10: Highly likely to significantly beat baseline
- 7-8: Likely to beat baseline with solid improvements
- 5-6: May beat baseline with moderate improvements
- 3-4: Uncertain if it will beat baseline
- 1-2: Unlikely to beat baseline
"""
        return prompt

    def parse_json_response(self, response_text):
        """Parse JSON từ response của LLM"""
        # Loại bỏ markdown code blocks
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*', '', response_text)
        response_text = response_text.strip()

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            if self.debug_mode:
                print(f"JSON parse error: {e}")
                print(f"Response: {response_text[:200]}...")
            return None

    def debate_and_get_advice(self, heuristic_info):
        """
        Main method: Nhận heuristic info và trả về advice sau debate

        Args:
            heuristic_info: dict chứa:
                - 'algorithm': str, mô tả thuật toán
                - 'code': str, code implementation
                - 'objective': float, objective value
                - (optional) 'other_inf': any additional info

        Returns:
            advice: dict chứa:
                - 'direction_type': str
                - 'rationale': str
                - 'key_aspects': list of str
                - 'expected_impact': str
                - 'vote_info': dict (votes, total_score, avg_score)
                - 'llm_name': str (tên LLM đề xuất direction này)
        """

        if self.debug_mode:
            print("\n" + "=" * 80)
            print("STARTING MULTI-LLM DEBATE")
            print(f"Number of advisor LLMs: {len(self.advisor_llms)}")
            print(f"Debate rounds: {self.debate_rounds}")
            print("=" * 80)

        # ===== ROUND 0: Initial Direction Proposals =====
        if self.debug_mode:
            print("\n>>> ROUND 0: Initial Proposals")

        current_directions = []
        for llm_info in self.advisor_llms:
            prompt = self.get_prompt_initial_direction(heuristic_info, llm_info)

            if self.debug_mode:
                print(f"\n--- {llm_info['name']} ({llm_info['role']}) proposing ---")

            response = llm_info['interface'].get_response(prompt)
            direction = self.parse_json_response(response)

            if direction:
                direction['llm_name'] = llm_info['name']
                direction['role'] = llm_info['role']
                direction['round'] = 0
                current_directions.append(direction)

                if self.debug_mode:
                    print(f"Reasoning: {direction.get('reasoning', 'N/A')[:80]}...")
                    print(f"Proposed: {direction.get('direction_type', 'N/A')}")
            else:
                if self.debug_mode:
                    print(f"Failed to parse from {llm_info['name']}")

        if len(current_directions) == 0:
            if self.debug_mode:
                print("ERROR: No valid directions generated!")
            return None

        # ===== DEBATE ROUNDS: Critique, Defend, Refine =====
        for round_num in range(1, self.debate_rounds + 1):
            if self.debug_mode:
                print(f"\n>>> ROUND {round_num}: Debate")

            refined_directions = []

            for i, llm_info in enumerate(self.advisor_llms):
                if i >= len(current_directions):
                    continue  # Nếu có ít hơn số LLM cố vấn
                own_direction = current_directions[i]
                other_directions = [d for j, d in enumerate(current_directions) if j != i]

                prompt = self.get_prompt_debate_round(
                    heuristic_info,
                    llm_info,
                    own_direction,
                    other_directions,
                    round_num
                )

                if self.debug_mode:
                    print(f"\n--- {llm_info['name']} debating ---")

                response = llm_info['interface'].get_response(prompt)
                debate_result = self.parse_json_response(response)

                if debate_result and 'refined_direction' in debate_result:
                    refined_dir = debate_result['refined_direction']
                    refined_dir['llm_name'] = llm_info['name']
                    refined_dir['role'] = llm_info['role']
                    refined_dir['round'] = round_num
                    refined_dir['reasoning'] = debate_result.get('reasoning', '')
                    refined_dir['critique'] = debate_result.get('critique_others', '')
                    refined_dir['defense'] = debate_result.get('defense', '')
                    refined_directions.append(refined_dir)

                    if self.debug_mode:
                        print(f"Reasoning: {refined_dir.get('reasoning', 'N/A')[:60]}...")
                        print(f"Refined to: {refined_dir.get('direction_type', 'N/A')}")
                else:
                    if self.debug_mode:
                        print(f"Parse failed, keeping previous")
                    refined_directions.append(own_direction)

            current_directions = refined_directions

        # ===== VOTING PHASE: Select Best Direction =====
        if self.debug_mode:
            print("\n>>> VOTING: Selecting best proposal")

        vote_results = {}
        for llm_info in self.advisor_llms:
            prompt = self.get_prompt_vote(current_directions, llm_info, heuristic_info)

            if self.debug_mode:
                print(f"\n--- {llm_info['name']} voting ---")

            response = llm_info['interface'].get_response(prompt)
            vote = self.parse_json_response(response)

            if vote and 'best_direction_llm' in vote:
                voted_llm = vote['best_direction_llm']
                score = vote.get('score', 5)

                if voted_llm not in vote_results:
                    vote_results[voted_llm] = {
                        'votes': 0,
                        'total_score': 0,
                        'reasons': []
                    }

                vote_results[voted_llm]['votes'] += 1
                vote_results[voted_llm]['total_score'] += score
                vote_results[voted_llm]['reasons'].append({
                    'reasoning': vote.get('reasoning', ''),
                    'why_best': vote.get('why_best', ''),
                    'weaknesses_to_address': vote.get('weaknesses_to_address', '')
                })

                if self.debug_mode:
                    print(f"Reasoning: {vote.get('reasoning', 'N/A')[:60]}...")
                    print(f"Voted for: {voted_llm} (score: {score})")
            else:
                if self.debug_mode:
                    print(f"Parse failed for {llm_info['name']}")

        # ===== SELECT BEST DIRECTION AND CREATE FINAL ADVICE =====
        if vote_results:
            # Chọn người chiến thắng dựa trên số phiếu + điểm trung bình
            best_llm_name = max(
                vote_results.items(),
                key=lambda x: (x[1]['votes'], x[1]['total_score'])
            )[0]

            best_direction = next(
                (d for d in current_directions if d['llm_name'] == best_llm_name),
                None
            )

            if best_direction:
                # Tạo final advice với "what to do" và "what to avoid"
                advice = {
                    'selected_direction': {
                        'type': best_direction['direction_type'],
                        'rationale': best_direction['rationale'],
                        'what_to_do': best_direction['key_aspects'],
                        'expected_impact': best_direction['expected_impact']
                    },
                    'what_to_avoid': [],
                    'vote_info': {
                        'winner': best_llm_name,
                        'votes': vote_results[best_llm_name]['votes'],
                        'total_score': vote_results[best_llm_name]['total_score'],
                        'avg_score': vote_results[best_llm_name]['total_score'] / vote_results[best_llm_name][
                            'votes'] if vote_results[best_llm_name]['votes'] > 0 else 0
                    },
                    'debate_summary': {
                        'total_rounds': self.debate_rounds,
                        'num_experts': len(self.advisor_llms)
                    }
                }

                # Thu thập "what to avoid" từ critiques của các direction bị loại
                for direction in current_directions:
                    if direction['llm_name'] != best_llm_name and 'critique' in direction:
                        if direction['critique']:
                            advice['what_to_avoid'].append({
                                'from': direction['llm_name'],
                                'critique': direction['critique']
                            })

                if self.debug_mode:
                    print(f"\n>>> FINAL DECISION: {best_llm_name}'s proposal selected")
                    print(f"Votes: {advice['vote_info']['votes']}/{len(self.advisor_llms)}")
                    print(f"Avg score: {advice['vote_info']['avg_score']:.2f}/10")
            else:
                advice = self._create_fallback_advice(current_directions[0])
        else:
            advice = self._create_fallback_advice(current_directions[0])

        if self.debug_mode:
            print("\n" + "=" * 80)
            print("DEBATE COMPLETED")
            print("=" * 80 + "\n")

        return advice

    def _create_fallback_advice(self, direction):
        """Tạo advice từ direction khi không có vote"""
        return {
            'selected_direction': {
                'type': direction['direction_type'],
                'rationale': direction['rationale'],
                'what_to_do': direction['key_aspects'],
                'expected_impact': direction.get('expected_impact', 'Unknown')
            },
            'what_to_avoid': [],
            'vote_info': {
                'winner': direction['llm_name'],
                'votes': 0,
                'total_score': 0,
                'avg_score': 0
            },
            'debate_summary': {
                'total_rounds': self.debate_rounds,
                'num_experts': len(self.advisor_llms)
            }
        }

    def get_advice(self, code, algorithm, objective):
        """
        Convenience method: Nhận code, algorithm, objective và trả về advice

        Args:
            code: str, code implementation
            algorithm: str, algorithm description
            objective: float, objective value

        Returns:
            advice: dict chứa direction từ debate
        """
        heuristic_info = {
            'code': code,
            'algorithm': algorithm,
            'objective': objective
        }

        return self.debate_and_get_advice(heuristic_info)