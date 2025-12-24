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

    def get_prompt_post(self, code, algorithm):

        prompt_content = self.prompt_task + "\n" + "Following is the a Code implementing a heuristic algorithm with function name " + self.prompt_func_name + " to solve the above mentioned problem.\n"
        prompt_content += self.prompt_inout_inf + " " + self.prompt_other_inf
        prompt_content += "\n\nCode:\n" + code
        prompt_content += "\n\nNow you should describe the Design Idea of the algorithm using less than 5 sentences.\n"
        prompt_content += "Hint: You should highlight every meaningful designs in the provided code and describe their ideas. You can analyse the code to see which variables are given higher values and which variables are given lower values, the choice of parameters or the total structure of the code."
        return prompt_content

    def get_prompt_refine(self, code, algorithm):

        prompt_content = self.prompt_task + "\n" + "Following is the Design Idea of a heuristic algorithm for the problem and the code with function name '" + self.prompt_func_name + "' for implementing the heuristic algorithm.\n"
        prompt_content += self.prompt_inout_inf + " " + self.prompt_other_inf
        prompt_content += "\nDesign Idea:\n" + algorithm
        prompt_content += "\n\nCode:\n" + code
        prompt_content += "\n\nThe content of the Design Idea idea cannot fully represent what the algorithm has done informative. So, now you should re-describe the algorithm using less than 3 sentences.\n"
        prompt_content += "Hint: You should reference the given Design Idea and highlight the most critical design ideas of the code. You can analyse the code to describe which variables are given higher priorities and which variables are given lower priorities, the parameters and the structure of the code."
        return prompt_content

    def get_prompt_i1(self):
        prompt_content = self.prompt_task + "\n" + "First, describe the design idea and main steps of your algorithm in one sentence. " + "The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations. Remember to import necessary modules and check the syntax error"
        return prompt_content

    def get_prompt_e1(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            # print(indivs[i]['algorithm'] + f"Objective value: {indivs[i]['objective']}")
            prompt_indiv = prompt_indiv + "No." + str(
                i + 1) + " algorithm's description, its corresponding code and its objective value are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i][
                               'code'] + "\n" + f"Objective value: {indivs[i]['objective']}" + "\n\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have " + str(
            len(indivs)) + " existing algorithms with their codes as follows: \n\n" \
                         + prompt_indiv + \
                         "Please create a new algorithm that has a totally different form from the given algorithms. Try generating codes with different structures, flows or algorithms. The new algorithm should have a relatively low objective value. \n" \
                         "First, describe the design idea and main steps of your algorithm in one sentence. The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations. Remember to import necessary modules and check the syntax error"
        return prompt_content

    def get_prompt_e2(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            # print(indivs[i]['algorithm'] + f"Objective value: {indivs[i]['objective']}")
            prompt_indiv = prompt_indiv + "No." + str(
                i + 1) + " algorithm's description, its corresponding code and its objective value are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i][
                               'code'] + "\n" + f"Objective value: {indivs[i]['objective']}" + "\n\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have " + str(
            len(indivs)) + " existing algorithms with their codes and objective values as follows: \n\n" \
                         + prompt_indiv + \
                         f"Please create a new algorithm that has a similar form to the No.{len(indivs)} algorithm and is inspired by the No.{1} algorithm. The new algorithm should have a objective value lower than both algorithms.\n" \
                         f"Firstly, list the common ideas in the No.{1} algorithm that may give good performances. Secondly, based on the common idea, describe the design idea based on the No.{len(indivs)} algorithm and main steps of your algorithm in one sentence. \
The description must be inside a brace. Thirdly, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations.  Remember to import necessary modules and check the syntax error"
        return prompt_content

    def get_prompt_m1(self, indiv1):
        prompt_content = self.prompt_task + "\n" \
                                            "I have one algorithm with its code as follows. \n\n\
Algorithm's description: " + indiv1['algorithm'] + "\n\
Code:\n\
" + indiv1['code'] + "\n\
Please create a new algorithm that has a different form but can be a modified version of the provided algorithm. Attempt to introduce more novel mechanisms and new equations or programme segments.\n" \
                     "First, describe the design idea based on the provided algorithm and main steps of the new algorithm in one sentence. \
The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations.  Remember to import necessary modules and check the syntax error"
        return prompt_content

    def get_prompt_m2(self, indiv1):
        prompt_content = self.prompt_task + "\n" \
                                            "I have one algorithm with its code as follows. \n\n\
Algorithm's description: " + indiv1['algorithm'] + "\n\
Code:\n\
" + indiv1['code'] + "\n\
Please identify the main algorithm parameters and help me in creating a new algorithm that has different parameter settings to equations compared to the provided algorithm. \n" \
                     "First, describe the design idea based on the provided algorithm and main steps of the new algorithm in one sentence. \
The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations. Remember to import necessary modules and check the syntax error"
        return prompt_content

    def get_prompt_s1(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = prompt_indiv + "No." + str(
                i + 1) + " algorithm's description, its corresponding code and its objective value are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i][
                               'code'] + "\n" + f"Objective value: {indivs[i]['objective']}" + "\n\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have " + str(
            len(indivs)) + " existing algorithms with their codes and objective values as follows: \n\n" \
                         + prompt_indiv + \
                         f"Please help me create a new algorithm that is inspired by all the above algorithms with its objective value lower than any of them.\n" \
                         "Firstly, list some ideas in the provided algorithms that are clearly helpful to a better algorithm. Secondly, based on the listed ideas, describe the design idea and main steps of your new algorithm in one sentence. \
The description must be inside a brace. Thirdly, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_counter(self, indiv1):
        prompt_content = self.prompt_task + "\n" \
                                            "I have one algorithm with its code as follows.\n\n" \
                                            "Algorithm's description: " + indiv1['algorithm'] + "\n" \
                                                                                                "Code:\n" + indiv1[
                             'code'] + "\n" \
                                       "Please analyze the provided algorithm carefully to identify any weaknesses, inefficiencies, or limitations in its design or implementation.\n" \
                                       "Then, create a new algorithm that specifically exploits these weaknesses to outperform or counter the original one.\n" \
                                       "Focus on areas where the opponent’s approach is suboptimal or vulnerable, and redesign or optimize those parts.\n" \
                                       "First, describe the design idea based on the provided algorithm and the main steps of the new algorithm in one sentence. " \
                                       "The description must be inside a brace outside the code implementation. " \
                                       "Next, implement it in Python as a function named '" + self.prompt_func_name + "'.\n" \
                                                                                                                      "This function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " + self.joined_inputs + ". " \
                                                                                 "The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " + self.joined_outputs + ". " \
                         + self.prompt_inout_inf + " " + self.prompt_other_inf + "\n" \
                                                                                 "Do not give additional explanations.  Remember to import necessary modules and check the syntax error"
        return prompt_content

    def counter(self, parents):

        prompt_content = self.get_prompt_counter(parents)

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

        # algorithm = response.split(':')[-1]
        return response

    def _get_alg(self, prompt_content):
        response = self.interface_llm.get_response(prompt_content)

        # SỬA: Kiểm tra match trước khi gọi group()
        match = re.search(r"\{(.*?)\}", response, re.DOTALL)
        if match:
            algorithm = match.group(1)
        else:
            algorithm = ""

        # Logic dự phòng (Fallback)
        if len(algorithm) == 0:
            if 'python' in response:
                # Tìm text trước từ khóa python
                algo_match = re.findall(r'^.*?(?=python)', response, re.DOTALL)
                algorithm = algo_match[0] if algo_match else ""
            elif 'import' in response:
                algo_match = re.findall(r'^.*?(?=import)', response, re.DOTALL)
                algorithm = algo_match[0] if algo_match else ""
            else:
                algo_match = re.findall(r'^.*?(?=def)', response, re.DOTALL)
                algorithm = algo_match[0] if algo_match else ""

        code = re.findall(r"import.*return", response, re.DOTALL)
        if len(code) == 0:
            code = re.findall(r"def.*return", response, re.DOTALL)

        n_retry = 1
        # SỬA: Áp dụng logic an toàn tương tự cho vòng lặp retry
        while (len(algorithm) == 0 or len(code) == 0):
            if self.debug_mode:
                print("Error: algorithm or code not identified, wait 1 seconds and retrying ... ")
            
            # Cần gọi lại LLM ở đây nếu muốn retry thật sự, nhưng code cũ chỉ parse lại response cũ? 
            # Giả sử logic cũ là muốn gọi lại response mới (nhưng code gốc bạn gửi lại dùng biến response cũ ở dòng 328, tôi giữ nguyên logic gọi API lại nếu cần)
            response = self.interface_llm.get_response(prompt_content) 

            match = re.search(r"\{(.*?)\}", response, re.DOTALL)
            if match:
                algorithm = match.group(1)
            else:
                algorithm = ""

            if len(algorithm) == 0:
                if 'python' in response:
                    algo_match = re.findall(r'^.*?(?=python)', response, re.DOTALL)
                    algorithm = algo_match[0] if algo_match else ""
                elif 'import' in response:
                    algo_match = re.findall(r'^.*?(?=import)', response, re.DOTALL)
                    algorithm = algo_match[0] if algo_match else ""
                else:
                    algo_match = re.findall(r'^.*?(?=def)', response, re.DOTALL)
                    algorithm = algo_match[0] if algo_match else ""

            code = re.findall(r"import.*return", response, re.DOTALL)
            if len(code) == 0:
                code = re.findall(r"def.*return", response, re.DOTALL)

            if n_retry > 3:
                break
            n_retry += 1

        # SỬA: Kiểm tra nếu code vẫn rỗng để tránh lỗi index out of range
        if len(code) == 0:
             # Trả về dummy code để tránh crash, hoặc raise error
             return ["def function(): return None", "Error parsing"]

        code = code[0]
        code_all = code + " " + ", ".join(s for s in self.prompt_func_outputs)

        return [code_all, algorithm]

    def post_thought(self, code, algorithm):

        prompt_content = self.get_prompt_refine(code, algorithm)

        post_thought = self._get_thought(prompt_content)

        return post_thought

    def i1(self):

        prompt_content = self.get_prompt_i1()

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

    def e1(self, parents):

        prompt_content = self.get_prompt_e1(parents)

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

    def e2(self, parents):

        prompt_content = self.get_prompt_e2(parents)

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

    def m1(self, parents):

        prompt_content = self.get_prompt_m1(parents)

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

    def m2(self, parents):

        prompt_content = self.get_prompt_m2(parents)

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

    def s1(self, parents):

        prompt_content = self.get_prompt_s1(parents)

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
    
    # [THÊM MỚI] Prompt cho Critic đánh giá Code
    def get_prompt_critic(self, code, algorithm, objective=None):
        prompt_content = self.prompt_task + "\n"
        prompt_content += "I have a heuristic algorithm with the following code:\n"
        prompt_content += code + "\n\n"
        if objective is not None:
            prompt_content += f"Its current objective value is: {objective}\n"
        
        prompt_content += "Act as a critical evaluator (Critic). Analyze the code and identify 3 key weaknesses or potential risks (e.g., local optima, high complexity, lack of diversity).\n"
        prompt_content += "Provide your feedback in a concise list. Do not generate new code yet."
        return prompt_content

    # Tìm hàm này và thay thế nội dung
    def get_prompt_refine_with_critic(self, code, critic_feedback):
        prompt_content = self.prompt_task + "\n"
        prompt_content += "I have a heuristic algorithm:\n" + code + "\n\n"
        prompt_content += "A Critic has provided the following feedback:\n" + critic_feedback + "\n\n"
        prompt_content += "Based on this feedback, please act as an Exploiter/Developer to refine the code.\n"
        
        # --- PHẦN THÊM VÀO ---
        prompt_content += "First, describe the design idea and main steps of your refined algorithm in one sentence. The description must be inside a brace outside the code implementation.\n"
        # ---------------------

        prompt_content += "1. Address the weaknesses mentioned.\n"
        prompt_content += "2. Keep the logic that works well.\n"
        # prompt_content += "3. Implement the improved version in Python as a function named '" + self.prompt_func_name + "'.\n"
        prompt_content +=  "Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations. Remember to import necessary modules and no syntax error"
        return prompt_content

    # [THÊM MỚI] Hàm gọi LLM cho Critic
    def critic(self, code, algorithm, objective=None):
        prompt = self.get_prompt_critic(code, algorithm, objective)
        if self.debug_mode:
            print("\n >>> Critic is analyzing...")
        response = self._get_thought(prompt) # Critic chỉ trả về text, không cần parse code
        return response

    # [THÊM MỚI] Hàm gọi LLM cho Refine
    def refine_with_critic(self, code, critic_feedback):
        prompt = self.get_prompt_refine_with_critic(code, critic_feedback)
        if self.debug_mode:
            print("\n >>> Exploiter is refining based on critic...")
        [code_all, algorithm] = self._get_alg(prompt)
        return [code_all, algorithm]
