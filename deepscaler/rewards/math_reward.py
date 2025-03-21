"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from typing import List, Union

from deepscaler.globals import THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END, OAI_RM_MODEL
from deepscaler.rewards import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from deepscaler.rewards.math_utils.utils import extract_answer, grade_answer_sympy, grade_answer_mathd
from deepscaler.system_prompts import ORM_PROMPT
from deepscaler.utils import call_gemini_llm, call_oai_rm_llm

ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""

class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput) -> RewardOutput:
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)
        
        problem = input.problem
        model_response = input.model_response
        
        # Extract solution.
        if THOUGHT_DELIMITER_START in model_response and THOUGHT_DELIMITER_END in model_response:
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        else:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
        
        model_answer = extract_answer(model_solution)
        if model_answer is None:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # Process the ground truth(s)
        ground_truths = input.ground_truth.get("answer", None)
        if ground_truths is None:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)
        
        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, (str, float, int)):
            ground_truths = [ground_truths]
            
        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)
        
        if not processed_ground_truths:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
            if is_correct:
                return RewardOutput(reward=self.config.correct_reward, is_correct=True)

        # If latex heuristics fail and ORM is enabled, use LLM as ORM to evaluate correctness
        if self.config.use_math_orm:
            for ground_truth in processed_ground_truths:
                try:
                    orm_response = call_gemini_llm(
                        system_prompt=ORM_PROMPT,
                        prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
                        temperature=0.0,
                    )

                    if "[[YES]]" in orm_response:
                        return RewardOutput(reward=self.config.correct_reward, is_correct=True)
                except Exception as e:
                    print ("Error calling Gemini ORM, trying OAI RM")
                    orm_response = call_oai_rm_llm(
                        system_prompt=ORM_PROMPT,
                        prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
                        temperature=0.0,
                        model_id=OAI_RM_MODEL,
                    )
                    
                    if "[[YES]]" in orm_response:
                        return RewardOutput(reward=self.config.correct_reward, is_correct=True)
                    continue
                
        return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)

class CustomRewardMathFn(RewardMathFn):
    """
    Custom reward function for evaluating mathematical answers with additional logic.
    """

    def compute_correctness_and_length(self, input: RewardInput) -> List[RewardOutput]:
        """
        计算所有 response 的正确性和长度。
        """
        results = []
        for response in input.responses:
            reward_output = super().__call__(response)
            results.append(reward_output)
        return results

    def compute_reward(self, input: RewardInput, L_budget: float, is_correct: bool, length: int) -> RewardOutput:
        """
        计算每个 response 的奖励。
        """
        # 计算 lambda
        L_i = length
        lambda_value = (L_i - L_budget) / L_budget if L_budget != 0 else 0
        
        # 根据 lambda 和正确性修改奖励
        if is_correct:
            reward = max(-0.5 * lambda_value + 0.5, 0.1)
        else:
            reward = max(0.9 * lambda_value - 0.1, -0.1)
        
        # 解析 <fast_think> 和 <slow_think> 内容
        fast_think_content = self.extract_think_content(input.model_response, "fast_think")
        slow_think_content = self.extract_think_content(input.model_response, "slow_think")
        
        # 计算快思考和慢思考的比例
        total_think_content = len(fast_think_content) + len(slow_think_content)
        fast_think_ratio = len(fast_think_content) / total_think_content if total_think_content > 0 else 0
        slow_think_ratio = len(slow_think_content) / total_think_content if total_think_content > 0 else 0
        
        # 根据 p 调整对 <fast_think> 和 <slow_think> 的奖励
        if p > 0.5:
            # p 大则鼓励快思考比例增加
            reward += 0.1 * fast_think_ratio
        else:
            # p 小则鼓励慢思考比例增加
            reward += 0.1 * slow_think_ratio
        
        return RewardOutput(reward=reward, is_correct=is_correct)
    
    def extract_think_content(self, response: str, think_type: str) -> str:
        """
        提取指定类型的思考内容（<fast_think> 或 <slow_think>）。
        """
        start_tag = f"<{think_type}>"
        end_tag = f"</{think_type}>"
        if start_tag in response and end_tag in response:
            return response.split(start_tag)[1].split(end_tag)[0]
        return ""

def deepscaler_reward_fn(solution_str: str, ground_truth: Union[str, List[str]], enable_llm = False):
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm
    reward_fn = CustomRewardMathFn(reward_config)
    reward_response = reward_fn.compute_correctness_and_length(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, ground_truth={"answer": ground_truth}))
    return reward_response.is_correct

if __name__ == "__main__":
    reward = RewardMathFn(RewardConfig)
    input = RewardInput(problem="Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$.", problem_type=RewardType.MATH, model_response="<think> I am omniscient. </think> The answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.", ground_truth={"answer": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"]})
    output = reward(input)
    print(output)