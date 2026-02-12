import re
import time
from .interface_LLM import InterfaceAPI as InterfaceLLM
import re

input = lambda: ...


class Evolution():

    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode, prompts, **kwargs):
        assert 'use_local_llm' in kwargs
        assert 'url' in kwargs
        self._use_local_llm = kwargs.get('use_local_llm')
        self._url = kwargs.get('url')
        # -----------------------------------------------------------

        # set prompt interface
        # getprompts = GetPrompts()
        self.prompt_task = prompts.get_task()
        self.prompt_func_name = prompts.get_func_name()
        self.prompt_func_inputs = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf = prompts.get_inout_inf()
        self.prompt_other_inf = prompts.get_other_inf()
        self.prompt_seed_code = prompts.get_seed_func()
        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"

        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"

        # set LLMs
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode  # close prompt checking

        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM, self.debug_mode)

    def _format_advice(self, advice):
        if advice is None:
            return ""

        # Nếu là dict → chuyển thành text có cấu trúc
        if isinstance(advice, dict):
            selected = advice.get('selected_direction', {})
            direction_type = selected.get('type', '').strip()
            rationale = selected.get('rationale', '').strip()
            what_to_do = selected.get('what_to_do', [])
            expected_impact = selected.get('expected_impact', '').strip()

            # Định dạng "what to do" thành danh sách
            what_to_do_str = "\n".join(f"- {item.strip()}" for item in what_to_do if item)

            parts = []
            if direction_type:
                parts.append(f"Improvement Direction: {direction_type}")
            if rationale:
                parts.append(f"Rationale: {rationale}")
            if what_to_do_str:
                parts.append(f"Implementation Guidance:\n{what_to_do_str}")
            if expected_impact:
                parts.append(f"Expected Impact: {expected_impact}")

            # (Tùy chọn) thêm "what to avoid"
            what_to_avoid = advice.get('what_to_avoid', [])
            if what_to_avoid:
                avoid_items = []
                for item in what_to_avoid:
                    if isinstance(item, dict):
                        critique = item.get('critique', '')
                        if critique:
                            avoid_items.append(critique)
                    elif isinstance(item, str) and item.strip():
                        avoid_items.append(item.strip())
                if avoid_items:
                    avoid_str = "\n".join(f"- {item}" for item in avoid_items)
                    parts.append(f"Cautions / What to Avoid:\n{avoid_str}")

            return "\n\n".join(parts) if parts else ""

        # Nếu là chuỗi → xử lý như cũ (tương thích)
        if isinstance(advice, str):
            return advice.strip()

        # Trường hợp fallback
        return str(advice).strip()

    def get_prompt_post(self, code, algorithm, advice=None):
        prompt_content = self.prompt_task + "\n" + "Following is the a Code implementing a heuristic algorithm with function name " + self.prompt_func_name + " to solve the above mentioned problem.\n"
        prompt_content += self.prompt_inout_inf + " " + self.prompt_other_inf
        prompt_content += "\n\nCode:\n" + code
        prompt_content += self._format_advice(advice)
        prompt_content += "\n\nNow you should describe the Design Idea of the algorithm using less than 5 sentences.\n"
        prompt_content += "Hint: You should highlight every meaningful designs in the provided code and describe their ideas. You can analyse the code to see which variables are given higher values and which variables are given lower values, the choice of parameters or the total structure of the code."
        return prompt_content

    def get_prompt_refine(self, code, algorithm, advice=None):
        prompt_content = self.prompt_task + "\n" + "Following is the Design Idea of a heuristic algorithm for the problem and the code with function name '" + self.prompt_func_name + "' for implementing the heuristic algorithm.\n"
        prompt_content += self.prompt_inout_inf + " " + self.prompt_other_inf
        prompt_content += "\nDesign Idea:\n" + algorithm
        prompt_content += "\n\nCode:\n" + code
        prompt_content += self._format_advice(advice)
        prompt_content += "\n\nThe content of the Design Idea idea cannot fully represent what the algorithm has done informative. So, now you should re-describe the algorithm using less than 3 sentences.\n"
        prompt_content += "Hint: You should reference the given Design Idea and highlight the most critical design ideas of the code. You can analyse the code to describe which variables are given higher priorities and which variables are given lower priorities, the parameters and the structure of the code."
        return prompt_content

    def get_prompt_i1(self, advice=None):
        prompt_content = self.prompt_task + "\n"
        prompt_content += self._format_advice(advice)
        prompt_content += "First, describe the design idea and main steps of your algorithm in one sentence. " + "The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_e1(self, indivs, advice=None):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = prompt_indiv + "No." + str(
                i + 1) + " algorithm's description, its corresponding code and its objective value are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i][
                               'code'] + "\n" + f"Objective value: {indivs[i]['objective']}" + "\n\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have " + str(
            len(indivs)) + " existing algorithms with their codes as follows: \n\n" \
                         + prompt_indiv
        prompt_content += self._format_advice(advice)
        prompt_content += "Please create a new algorithm that has a totally different form from the given algorithms. Try generating codes with different structures, flows or algorithms. The new algorithm should have a relatively low objective value. \n" \
                         "First, describe the design idea and main steps of your algorithm in one sentence. The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations. Use only standard English letters and numbers. Do not output any special characters."
        return prompt_content

    def get_prompt_e2(self, indivs, advice=None):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = prompt_indiv + "No." + str(
                i + 1) + " algorithm's description, its corresponding code and its objective value are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i][
                               'code'] + "\n" + f"Objective value: {indivs[i]['objective']}" + "\n\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have " + str(
            len(indivs)) + " existing algorithms with their codes and objective values as follows: \n\n" \
                         + prompt_indiv
        prompt_content += self._format_advice(advice)
        prompt_content += f"Please create a new algorithm that has a similar form to the No.{len(indivs)} algorithm and is inspired by the No.{1} algorithm. The new algorithm should have a objective value lower than both algorithms.\n" \
                         f"Firstly, list the common ideas in the No.{1} algorithm that may give good performances. Secondly, based on the common idea, describe the design idea based on the No.{len(indivs)} algorithm and main steps of your algorithm in one sentence. \
The description must be inside a brace. Thirdly, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations. Use only standard English letters and numbers. Do not output any special characters."
        return prompt_content

    def get_prompt_m1(self, indiv1, advice=None):
        prompt_content = self.prompt_task + "\n" \
                                            "I have one algorithm with its code as follows. \n\n\
Algorithm's description: " + indiv1['algorithm'] + "\n\
Code:\n\
" + indiv1['code'] + "\n"
        prompt_content += self._format_advice(advice)
        prompt_content += "Please create a new algorithm that has a different form but can be a modified version of the provided algorithm. Attempt to introduce more novel mechanisms and new equations or programme segments.\n" \
                     "First, describe the design idea based on the provided algorithm and main steps of the new algorithm in one sentence. \
The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations. Use only standard English letters and numbers. Do not output any special characters."
        return prompt_content

    def get_prompt_m2(self, indiv1, advice=None):
        prompt_content = self.prompt_task + "\n" \
                                            "I have one algorithm with its code as follows. \n\n\
Algorithm's description: " + indiv1['algorithm'] + "\n\
Code:\n\
" + indiv1['code'] + "\n"
        prompt_content += self._format_advice(advice)
        prompt_content += "Please identify the main algorithm parameters and help me in creating a new algorithm that has different parameter settings to equations compared to the provided algorithm. \n" \
                     "First, describe the design idea based on the provided algorithm and main steps of the new algorithm in one sentence. \
The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations. Use only standard English letters and numbers. Do not output any special characters."
        return prompt_content

    def get_prompt_s1(self, indivs, advice=None):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = prompt_indiv + "No." + str(
                i + 1) + " algorithm's description, its corresponding code and its objective value are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i][
                               'code'] + "\n" + f"Objective value: {indivs[i]['objective']}" + "\n\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have " + str(
            len(indivs)) + " existing algorithms with their codes and objective values as follows: \n\n" \
                         + prompt_indiv
        prompt_content += self._format_advice(advice)
        prompt_content += f"Please help me create a new algorithm that is inspired by all the above algorithms with its objective value lower than any of them.\n" \
                         "Firstly, list some ideas in the provided algorithms that are clearly helpful to a better algorithm. Secondly, based on the listed ideas, describe the design idea and main steps of your new algorithm in one sentence. \
The description must be inside a brace. Thirdly, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations. Use only standard English letters and numbers. Do not output any special characters."
        return prompt_content

    def get_prompt_counter(self, indiv1, advice=None):
        prompt_content = self.prompt_task + "\n" \
                                            "I have one algorithm with its code as follows.\n\n" \
                                            "Algorithm's description: " + indiv1['algorithm'] + "\n" \
                                                                                                "Code:\n" + indiv1[
                             'code'] + "\n"
        prompt_content += self._format_advice(advice)
        prompt_content += "Please analyze the provided algorithm carefully to identify any weaknesses, inefficiencies, or limitations in its design or implementation.\n" \
                                       "Then, create a new algorithm that specifically exploits these weaknesses to outperform or counter the original one.\n" \
                                       "Focus on areas where the opponent's approach is suboptimal or vulnerable, and redesign or optimize those parts.\n" \
                                       "First, describe the design idea based on the provided algorithm and the main steps of the new algorithm in one sentence. " \
                                       "The description must be inside a brace outside the code implementation. " \
                                       "Next, implement it in Python as a function named '" + self.prompt_func_name + "'.\n" \
                                                                                                                      "This function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " + self.joined_inputs + ". " \
                                                                                 "The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " + self.joined_outputs + ". " \
                         + self.prompt_inout_inf + " " + self.prompt_other_inf + "\n" \
                                                                                 "Do not give additional explanations. Use only standard English letters and numbers. Do not output any special characters."
        return prompt_content

    def counter(self, parents, advice=None):
        prompt_content = self.get_prompt_counter(parents, advice)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ counter ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def _get_thought(self, prompt_content):
        response = self.interface_llm.get_response(prompt_content)
        return response

    def _get_alg(self, prompt_content):
        response = self.interface_llm.get_response(prompt_content)

        algorithm = re.search(r"\{(.*?)\}", response, re.DOTALL).group(1)
        if len(algorithm) == 0:
            if 'python' in response:
                algorithm = re.findall(r'^.*?(?=python)', response, re.DOTALL)
            elif 'import' in response:
                algorithm = re.findall(r'^.*?(?=import)', response, re.DOTALL)
            else:
                algorithm = re.findall(r'^.*?(?=def)', response, re.DOTALL)

        code = re.findall(r"import.*return", response, re.DOTALL)
        if len(code) == 0:
            code = re.findall(r"def.*return", response, re.DOTALL)

        n_retry = 1
        while (len(algorithm) == 0 or len(code) == 0):
            if self.debug_mode:
                print("Error: algorithm or code not identified, wait 1 seconds and retrying ... ")

            response = self.interface_llm.get_response(prompt_content)

            algorithm = re.search(r"\{(.*?)\}", response, re.DOTALL).group(1)
            if len(algorithm) == 0:
                if 'python' in response:
                    algorithm = re.findall(r'^.*?(?=python)', response, re.DOTALL)
                elif 'import' in response:
                    algorithm = re.findall(r'^.*?(?=import)', response, re.DOTALL)
                else:
                    algorithm = re.findall(r'^.*?(?=def)', response, re.DOTALL)

            code = re.findall(r"import.*return", response, re.DOTALL)
            if len(code) == 0:
                code = re.findall(r"def.*return", response, re.DOTALL)

            if n_retry > 3:
                break
            n_retry += 1

        code = code[0]
        code_all = code + " " + ", ".join(s for s in self.prompt_func_outputs)

        return [code_all, algorithm]

    def post_thought(self, code, algorithm, advice=None):
        prompt_content = self.get_prompt_refine(code, algorithm, advice)
        post_thought = self._get_thought(prompt_content)
        return post_thought

    def i1(self, advice=None):
        prompt_content = self.get_prompt_i1(advice)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def e1(self, parents, advice=None):
        prompt_content = self.get_prompt_e1(parents, advice)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def e2(self, parents, advice=None):
        prompt_content = self.get_prompt_e2(parents, advice)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def m1(self, parents, advice=None):
        prompt_content = self.get_prompt_m1(parents, advice)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def m2(self, parents, advice=None):
        prompt_content = self.get_prompt_m2(parents, advice)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def s1(self, parents, advice=None):
        prompt_content = self.get_prompt_s1(parents, advice)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ s1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def get_prompt_error_signal(self, indiv):
        prompt = (
                self.prompt_task + "\n\n"
                                   "You are given an algorithm and its implementation.\n\n"
                                   "Algorithm description:\n"
                + indiv['algorithm'] + "\n\n"
                                       "Code:\n"
                + indiv['code'] + "\n\n"
                                  "Your task is NOT to judge correctness.\n"
                                  "Your task is to surface *subtle discomforts* or *uneasy feelings* about this algorithm.\n\n"
                                  "Think like a human expert who says:\n"
                                  "\"I can't prove it's wrong yet, but something here feels fragile or over-assumed.\"\n\n"
                                  "Focus on:\n"
                                  "- Implicit assumptions that may not always hold\n"
                                  "- Design choices that feel brittle or overly confident\n"
                                  "- Parts that might fail silently rather than obviously\n\n"
                                  "Strict rules:\n"
                                  "- Do NOT propose solutions\n"
                                  "- Do NOT suggest fixes\n"
                                  "- Do NOT rewrite code\n"
                                  "- Do NOT explain in detail\n\n"
                                  "List 3–5 brief uncertainty signals (one line each):\n"
                                  "- "
        )
        return prompt

    def get_prompt_counterfactual(self, indiv, error_signal):
        prompt = (
                self.prompt_task + "\n\n"
                                   "You are given an algorithm and its implementation.\n\n"
                                   "Algorithm description:\n"
                + indiv['algorithm'] + "\n\n"
                                       "Code:\n"
                + indiv['code'] + "\n\n"
                                  "Previously identified uncertainty signals:\n"
                + error_signal + "\n\n"
                                 "Now assume the algorithm's behavior is WRONG.\n\n"
                                 "Your goal is to imagine the *simplest possible situation* where this algorithm fails.\n"
                                 "The failure should come from assumptions breaking, not from extreme or unrealistic cases.\n\n"
                                 "Strict rules:\n"
                                 "- Do NOT defend the algorithm\n"
                                 "- Do NOT propose fixes\n"
                                 "- Do NOT modify the algorithm\n\n"
                                 "Describe ONE minimal failure case:\n"
                                 "- Scenario: (brief, concrete)\n"
                                 "- Why this breaks the algorithm: (conceptual reason)\n"
        )
        return prompt

    def get_prompt_role_conflict(self, indiv, counterfactual):
        prompt = (
                self.prompt_task + "\n\n"
                                   "You are given an algorithm and its implementation.\n\n"
                                   "Algorithm description:\n"
                + indiv['algorithm'] + "\n\n"
                                       "Code:\n"
                + indiv['code'] + "\n\n"
                                  "Identified failure scenario:\n"
                + counterfactual + "\n\n"
                                   "You will analyze this algorithm from three strictly separated roles.\n\n"
                                   "ROLE 1 — Defender:\n"
                                   "Argue why the original design choices are reasonable under the intended assumptions.\n"
                                   "Do NOT address the counterfactual directly.\n\n"
                                   "ROLE 2 — Attacker:\n"
                                   "Argue why the counterfactual exposes a fundamental weakness.\n"
                                   "Do NOT suggest solutions.\n\n"
                                   "ROLE 3 — Observer:\n"
                                   "Summarize the *core unresolved tension* between Defender and Attacker.\n"
                                   "Do NOT choose a winner.\n"
                                   "Do NOT propose fixes.\n\n"
                                   "Output:\n"
                                   "Defender:\n"
                                   "- ...\n\n"
                                   "Attacker:\n"
                                   "- ...\n\n"
                                   "Observer (conflicts only):\n"
                                   "- ...\n"
        )
        return prompt

    def get_prompt_abstraction(self, role_conflict):
        prompt = (
                self.prompt_task + "\n\n"
                                   "Below is a multi-perspective analysis of an algorithm.\n\n"
                                   "Analysis:\n"
                + role_conflict + "\n\n"
                                  "Your task is to step back and reason at a higher level of abstraction.\n\n"
                                  "Ignore all implementation details.\n"
                                  "Think in terms of *strategy*, *assumptions*, and *trade-offs*.\n\n"
                                  "Output format:\n"
                                  "Core strategy (1–2 bullets):\n"
                                  "- ...\n\n"
                                  "Hidden or risky assumptions:\n"
                                  "- ...\n"
                                  "- ...\n"
        )
        return prompt

    def get_prompt_assumption_repair(self, abstraction):
        prompt = (
                self.prompt_task + "\n\n"
                                   "Below are abstract principles and risky assumptions of an algorithm.\n\n"
                + abstraction + "\n\n"
                                "Your task is to revise ONLY the assumptions or applicability conditions.\n\n"
                                "You are NOT allowed to:\n"
                                "- Design a new algorithm\n"
                                "- Suggest new mechanisms\n"
                                "- Change the overall strategy\n\n"
                                "Only clarify, restrict, or condition the assumptions to make failures less likely.\n\n"
                                "Revised assumptions:\n"
                                "- ...\n"
                                "- ...\n"
        )
        return prompt

    def get_prompt_final_advice(self, repaired_assumptions):
        prompt = (
            self.prompt_task + "\n\n"
            "You are given revised assumptions that improve the robustness of an algorithm.\n\n"
            "Revised assumptions:\n"
            + repaired_assumptions + "\n\n"
            "Based ONLY on these assumptions, provide actionable guidance for improving the algorithm.\n\n"
            "Output in the following structured format:\n\n"
            "Improvement Direction: [short, concrete label]\n"
            "Rationale: [why adjusting the algorithm under these assumptions helps]\n"
            "Implementation Guidance:\n"
            "- ...\n"
            "- ...\n"
            "Expected Impact: [robustness / stability / generalization gain]\n\n"
            "Cautions / What to Avoid:\n"
            "- ...\n"
            "- ...\n"
        )
        return prompt

    def error_signal(self, indiv):
        prompt_content = self.get_prompt_error_signal(indiv)

        if self.debug_mode:
            print("\n >>> check prompt for [ error_signal ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        error_signal = self._get_thought(prompt_content)

        if self.debug_mode:
            print("\n >>> error signals: \n", error_signal)
            print(">>> Press 'Enter' to continue")
            input()

        return error_signal

    def counterfactual(self, indiv, error_signal):
        prompt_content = self.get_prompt_counterfactual(indiv, error_signal)

        if self.debug_mode:
            print("\n >>> check prompt for [ counterfactual ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        counterfactual = self._get_thought(prompt_content)

        if self.debug_mode:
            print("\n >>> counterfactual: \n", counterfactual)
            print(">>> Press 'Enter' to continue")
            input()

        return counterfactual

    def role_conflict(self, indiv, counterfactual):
        prompt_content = self.get_prompt_role_conflict(indiv, counterfactual)

        if self.debug_mode:
            print("\n >>> check prompt for [ role_conflict ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        conflict = self._get_thought(prompt_content)

        if self.debug_mode:
            print("\n >>> role conflict analysis: \n", conflict)
            print(">>> Press 'Enter' to continue")
            input()

        return conflict

    def abstraction(self, role_conflict):
        prompt_content = self.get_prompt_abstraction(role_conflict)

        if self.debug_mode:
            print("\n >>> check prompt for [ abstraction ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        abstraction = self._get_thought(prompt_content)

        if self.debug_mode:
            print("\n >>> abstract principles: \n", abstraction)
            print(">>> Press 'Enter' to continue")
            input()

        return abstraction

    def assumption_repair(self, abstraction):
        prompt_content = self.get_prompt_assumption_repair(abstraction)

        if self.debug_mode:
            print("\n >>> check prompt for [ assumption_repair ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        repaired = self._get_thought(prompt_content)

        if self.debug_mode:
            print("\n >>> repaired assumptions: \n", repaired)
            print(">>> Press 'Enter' to continue")
            input()

        return repaired

    def final_advice(self, repaired_assumptions):
        prompt_content = self.get_prompt_final_advice(repaired_assumptions)

        if self.debug_mode:
            print("\n >>> check prompt for [ final_advice ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        advice = self._get_thought(prompt_content)

        if self.debug_mode:
            print("\n >>> final advice: \n", advice)
            print(">>> Press 'Enter' to continue")
            input()

        return advice

    def ecdrr(self, indiv):
        """
        Run Error-Driven Counterfactual Role Reflection (EDCRR) as a 5-step pipeline.
        Returns: advice (str or dict) compatible with other Evolution methods.
        """
        # --- Step 0: Error Signal ---
        error_signal_output = self.error_signal(indiv)
        # print("Error Signal Output:", error_signal_output)
        # --- Step 1: Counterfactual Attack ---
        counterfactual_output = self.counterfactual(indiv, error_signal_output)
        # print("Counterfactual Output:", counterfactual_output)
        # --- Step 2: Role Conflict ---
        role_conflict_output = self.role_conflict(indiv, counterfactual_output)
        # print("Role Conflict Output:", role_conflict_output)
        # --- Step 3: Abstraction Shift ---
        abstraction_output = self.abstraction(role_conflict_output)
        # print("Abstraction Output:", abstraction_output)
        # --- Step 4: Assumption Repair ---
        repaired_assumptions = self.assumption_repair(abstraction_output)
        # print("Repaired Assumptions:", repaired_assumptions)
        # --- Step 5: Final Advice ---
        advice = self.final_advice(repaired_assumptions)
        # print("Final Advice:", advice)
        return advice

    def get_prompt_refine_with_critic(self, indiv, advice):

        prompt_content = self.prompt_task + "\n"

        prompt_content += (
                "You are an expert algorithm engineer and software developer with extensive experience in refining "
                "heuristic and optimization algorithms for reliability and performance. "
                "You are given a heuristic algorithm and its current Python implementation.\n\n"

                "Current code:\n"
                + indiv['code'] + "\n\n"
                         "A Critic has analyzed this implementation and produced the following feedback:\n"
                + advice + "\n\n"
                                    "Your task is to performe a refinement step by carefully revising "
                                    "the existing implementation to address the weaknesses identified by the Critic, while preserving "
                                    "the parts of the logic that already work well. "
                                    "You must improve robustness, correctness, clarity, and efficiency without changing the core "
                                    "algorithmic idea, the intended behavior, or the input–output interface.\n\n"

                                    "First, briefly describe the refined algorithm’s design idea and main steps in one sentence; "
                                    "this description must be enclosed in braces and placed outside the code implementation. "
                                    "Then, implement the refined version in Python as a function named '"
                + self.prompt_func_name + "'. "
                                          "The function should accept "
                + str(len(self.prompt_func_inputs)) + " input(s): "
                + self.joined_inputs + ". "
                + self.prompt_inout_inf + " "
                + self.prompt_other_inf + " "
                                          "Do not provide any additional explanations outside the required description and code. "
                                          "Ensure all necessary imports are included and that the code has no syntax errors."
        )

        return prompt_content

    def refine(self, indiv):

        advice = self.ecdrr(indiv)
        prompt_content = self.get_prompt_refine_with_critic(indiv, advice)

        if self.debug_mode:
            print("\n >>> check prompt for refining algorithm using [ refine_with_critic ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check refined algorithm: \n", algorithm)
            print("\n >>> check refined code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]



