import os
import time
import json
import re
from dotenv import load_dotenv
from openai import OpenAI
# Tenacity is used for automatic retries on API calls
from tenacity import retry, stop_after_attempt, wait_exponential

# --- Environment and API Setup ---
load_dotenv()

# API configuration - Read from environment variables
# NOTE: Ensure these variables are set in your .env file
# api_key = os.getenv("ZHIPU_API_KEY") # Example if using Zhipu
api_key = os.getenv("OPENAI_API_KEY") # Use standard OpenAI key name or specific one like ZHIPU_API_KEY
base_url = os.getenv("BASE_URL", "https://open.bigmodel.cn/api/paas/v4/") # Default to Zhipu endpoint if not set
chat_model = os.getenv("CHAT_MODEL", "glm-4-flash") # Default to specific model if not set

# Check if the essential API key is provided
if not api_key:
    raise ValueError("API key not found. Please set OPENAI_API_KEY (or the relevant key like ZHIPU_API_KEY) in your .env file.")

# Initialize the OpenAI client (works with compatible APIs)
client = OpenAI(
    api_key = api_key,
    base_url = base_url
)

# --- Utility Functions ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_completion(messages):
    """Calls the LLM API with retries on failure.

    Uses tenacity to retry up to 3 times with exponential backoff if the
    API call fails.

    Args:
        messages (list): A list of message dictionaries for the chat completion.

    Returns:
        str: The content of the LLM's response message.

    Raises:
        Exception: If the API call fails after all retries.
    """
    try:
        response = client.chat.completions.create(
            model=chat_model,
            messages=messages,
            # Consider adding temperature control if needed
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API call failed: {str(e)}")
        raise # Re-raise the exception after logging

def extract_json_content(text):
    """Extracts JSON content embedded within ```json ... ``` code blocks
       in the LLM's output text.

    Args:
        text (str): The raw text output from the LLM.

    Returns:
        str: The extracted JSON string, or the original text if no JSON block is found.
    """
    # Remove potential newlines that might interfere with regex
    text = text.replace("\n", "")
    # Regex to find content within ```json ``` blocks
    pattern = r"```json(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        # Return the first match, stripped of leading/trailing whitespace
        return matches[0].strip()
    # If no JSON block found, return the original text
    print(f"[JSON Parsing] No ```json``` block found in text: {text[:100]}...")
    return text

class JsonOutputParser:
    """Parses a string assumed to contain JSON into a Python dictionary."""
    def parse(self, result):
        """Parses the input string (potentially after extracting from code blocks).

        Args:
            result (str): The string potentially containing JSON.

        Returns:
            dict: The parsed Python dictionary.

        Raises:
            Exception: If the string is not valid JSON.
        """
        try:
            # First, try to extract content if it's wrapped in ```json ```
            json_string = extract_json_content(result)
            # Attempt to parse the extracted (or original) string as JSON
            parsed_result = json.loads(json_string)
            return parsed_result
        except json.JSONDecodeError as e:
            # Raise a more informative error if JSON parsing fails
            raise Exception(f"Invalid JSON output after extraction: '{json_string}'") from e

# --- Grading Agent Class ---
class GradingAssistant:
    """An agent that uses an LLM to grade student answers based on provided criteria.

    It takes a question, standard answer, grading criteria, and student answer,
    prompts the LLM to perform the grading, and parses the structured JSON output.
    """
    def __init__(self):
        """Initializes the GradingAssistant with a JSON parser and the system prompt."""
        self.output_parser = JsonOutputParser()

        # System prompt defining the LLM's role as a professional grading teacher
        # and specifying the required JSON output format.
        self.system_prompt = """你是一个专业的阅卷老师。你需要根据标准答案和评分标准，对学生的答案进行评分和点评。
评分时请注意以下几点：
1. 严格按照评分标准进行评分
2. 注意考察学生是否理解了题目的核心概念
3. 关注答案的完整性和准确性
4. 给出具体的得分点分析
5. 提供建设性的改进建议

你需要以JSON格式输出评阅结果，格式如下：
```json
{
    "score": 分数 (number),
    "analysis": {
        "points_earned": "得分点分析 (string)",
        "points_missed": "失分点分析 (string)",
        "suggestions": "改进建议 (string)"
    }
}
```

注意：
1. JSON 结构必须严格遵守，所有字符串值必须用双引号。
2. 分析和建议内容应简洁明了。
3. 分数应为数字类型。
"""

    def grade_answer(self, question, standard_answer, grading_criteria, student_answer):
        """Grades a single student answer using the LLM.

        Args:
            question (str): The question text.
            standard_answer (str): The standard/correct answer.
            grading_criteria (str): The criteria for grading.
            student_answer (str): The student's answer to be graded.

        Returns:
            dict: A dictionary containing the grading result (score, analysis).

        Raises:
            Exception: If the LLM call fails or the output cannot be parsed as JSON.
        """
        # Construct the user prompt containing all necessary information for the LLM
        prompt = f"""
题目：{question}

标准答案：{standard_answer}

评分标准：{grading_criteria}

学生答案：{student_answer}

请对这个答案进行评分和点评，并严格按照指定的JSON格式输出结果。
"""

        # Prepare the messages for the API call
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Call the LLM API (with built-in retries via get_completion)
        response = get_completion(messages)

        # Parse the LLM's response, expecting JSON output
        result = self.output_parser.parse(response)

        return result # Return the parsed dictionary

    def batch_grade(self, questions, standard_answers, grading_criterias, student_answers):
        """Grades multiple student answers in a batch.

        Args:
            questions (list): List of questions.
            standard_answers (list): List of standard answers.
            grading_criterias (list): List of grading criteria.
            student_answers (list): List of student answers.

        Returns:
            list: A list of dictionaries, each containing the grading result
                  for the corresponding question, or an error dictionary if grading failed.
        """
        results = []
        # Iterate through all provided questions and answers
        for i, (q, sa, gc, sta) in enumerate(zip(questions, standard_answers, grading_criterias, student_answers)):
            print(f"\nGrading question {i+1}...")
            try:
                # Grade each answer individually
                result = self.grade_answer(q, sa, gc, sta)
                results.append(result)
                # Optional: Add a small delay between API calls to avoid rate limits
                time.sleep(0.5)
            except Exception as e:
                # If grading fails for one item, record the error and continue
                print(f"Error grading question {i+1}: {str(e)}")
                results.append({
                    "question_index": i+1,
                    "error": f"Failed to grade: {str(e)}"
                 })
        return results

# --- Main Execution Block (Example Usage) ---
def main():
    """Demonstrates how to use the GradingAssistant."""
    # Create an instance of the grading agent
    grader = GradingAssistant()

    # Example data for a single grading task
    question = "请解释什么是面向对象编程，并说明其三大特性。"

    standard_answer = """面向对象编程是一种编程范式，它使用"对象"来封装数据和操作数据的方法。
其三大特性是：
1. 封装：将数据和方法封装在对象内部，对外只暴露必要的接口
2. 继承：子类可以继承父类的属性和方法，实现代码重用
3. 多态：同一个方法可以在不同对象上产生不同的行为"""

    grading_criteria = """总分10分
1. 正确解释面向对象编程的概念（4分）
2. 完整说明三大特性（每个特性2分，共6分）
- 封装（2分）
- 继承（2分）
- 多态（2分）"""

    student_answer = """面向对象编程就是用对象来编程，可以把数据封装起来。
它的特性有：
1. 封装：把数据藏起来
2. 继承：可以继承别的类
3. 多态：一个方法有多种形态"""

    try:
        print("Starting single answer grading...")
        # Call the grading method
        result = grader.grade_answer(question, standard_answer, grading_criteria, student_answer)
        print("\nGrading Result:")
        # Print the result nicely formatted
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"\nAn error occurred during grading: {str(e)}")

    # Example for batch grading (optional demonstration)
    # questions_batch = [question, "另一题..."]
    # answers_batch = [standard_answer, "另一题答案..."]
    # criteria_batch = [grading_criteria, "另一题标准..."]
    # student_answers_batch = [student_answer, "另一学生答案..."]
    # print("\nStarting batch grading...")
    # batch_results = grader.batch_grade(questions_batch, answers_batch, criteria_batch, student_answers_batch)
    # print("\nBatch Grading Results:")
    # print(json.dumps(batch_results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()