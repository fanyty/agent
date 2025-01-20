import os
import time
import json
import re
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# 加载环境变量
load_dotenv()

# API配置
api_key = "f719be31ebe94a14afcf9b331de55e08.N0UkfhMEUoOjtW53"
base_url = "https://open.bigmodel.cn/api/paas/v4/"
chat_model = "glm-4-flash"

# 构造client
client = OpenAI(
    api_key = api_key,
    base_url = base_url
)

def get_completion(messages):
    """
    调用智谱API的统一接口
    """
    try:
        response = client.chat.completions.create(
            model=chat_model,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API调用出错: {str(e)}")
        raise

def extract_json_content(text):
    # 提取大模型输出内容中的json部分
    text = text.replace("\n","")
    pattern = r"```json(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text

class JsonOutputParser:
    def parse(self, result):
        try:
            result = extract_json_content(result)
            parsed_result = json.loads(result)
            return parsed_result
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid json output: {result}") from e

class GradingAssistant:
    def __init__(self):
        self.output_parser = JsonOutputParser()
        
        # 系统提示词
        self.system_prompt = """你是一个专业的阅卷老师。你需要根据标准答案和评分标准，对学生的答案进行评分和点评。
评分时请注意以下几点：
1. 严格按照评分标准进行评分
2. 注意考察学生是否理解了题目的核心概念
3. 关注答案的完整性和准确性
4. 给出具体的得分点分析
5. 提供建设性的改进建议

你需要以JSON格式输出评阅结果，格式如下：
{
    "score": 分数,
    "analysis": {
        "points_earned": "得分点分析",
        "points_missed": "失分点分析",
        "suggestions": "改进建议"
    }
}

注意：
1. 所有文本内容需要用双引号包裹
2. 文本内容中不要包含换行符
3. 评语要简明扼要，直击重点
"""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def grade_answer(self, question, standard_answer, grading_criteria, student_answer):
        """
        评阅学生答案
        
        参数：
        question: 题目内容
        standard_answer: 标准答案
        grading_criteria: 评分标准
        student_answer: 学生答案
        
        返回：
        评分结果和评语（JSON格式）
        """
        prompt = f"""
题目：{question}

标准答案：{standard_answer}

评分标准：{grading_criteria}

学生答案：{student_answer}

请对这个答案进行评分和点评，并以JSON格式输出结果。
"""
        success = False
        while not success:
            try:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ]
                response = get_completion(messages)
                result = self.output_parser.parse(response)
                success = True
            except Exception as e:
                print(f"评分过程出错: {str(e)}")
                print("重试中...")
                continue
        
        return result

    def batch_grade(self, questions, standard_answers, grading_criterias, student_answers):
        """
        批量评阅多道题目
        
        参数：
        questions: 题目列表
        standard_answers: 标准答案列表
        grading_criterias: 评分标准列表
        student_answers: 学生答案列表
        
        返回：
        所有题目的评分结果和评语
        """
        results = []
        for i, (q, sa, gc, sta) in enumerate(zip(questions, standard_answers, grading_criterias, student_answers)):
            print(f"\n正在评阅第{i+1}题...")
            try:
                result = self.grade_answer(q, sa, gc, sta)
                results.append(result)
                # 添加短暂延迟，避免API调用过于频繁
                time.sleep(1)
            except Exception as e:
                print(f"第{i+1}题评阅失败: {str(e)}")
                results.append({"error": str(e)})
        return results

def main():
    # 示例使用
    grader = GradingAssistant()
    
    # 示例题目
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
        print("开始评阅...")
        result = grader.grade_answer(question, standard_answer, grading_criteria, student_answer)
        print("\n评阅结果：")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"程序执行出错: {str(e)}")

if __name__ == "__main__":
    main() 