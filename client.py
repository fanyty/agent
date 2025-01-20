import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 从环境变量中读取配置
api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('BASE_URL')
chat_model = os.getenv('CHAT_MODEL')

# 构造client
client = OpenAI(
    api_key = api_key,
    base_url = base_url
)

# 系统提示词
sys_prompt = """你是一个聪明的客服。您将能够根据用户的问题将不同的任务分配给不同的人。您有以下业务线：
1.用户注册。如果用户想要执行这样的操作，您应该发送一个带有"registered workers"的特殊令牌。并告诉用户您正在调用它。
2.用户数据查询。如果用户想要执行这样的操作，您应该发送一个带有"query workers"的特殊令牌。并告诉用户您正在调用它。
3.删除用户数据。如果用户想执行这种类型的操作，您应该发送一个带有"delete workers"的特殊令牌。并告诉用户您正在调用它。
"""

registered_prompt = """
您的任务是根据用户信息存储数据。您需要从用户那里获得以下信息：
1.用户名、性别、年龄
2.用户设置的密码
3.用户的电子邮件地址
如果用户没有提供此信息，您需要提示用户提供。如果用户提供了此信息，则需要将此信息存储在数据库中，并告诉用户注册成功。
存储方法是使用SQL语句。您可以使用SQL编写插入语句，并且需要生成用户ID并将其返回给用户。
如果用户没有新问题，您应该回复带有 "customer service" 的特殊令牌，以结束任务。
"""

query_prompt = """
您的任务是查询用户信息。您需要从用户那里获得以下信息：
1.用户ID
2.用户设置的密码
如果用户没有提供此信息，则需要提示用户提供。如果用户提供了此信息，那么需要查询数据库。如果用户ID和密码匹配，则需要返回用户的信息。
如果用户没有新问题，您应该回复带有 "customer service" 的特殊令牌，以结束任务。
"""

delete_prompt = """
您的任务是删除用户信息。您需要从用户那里获得以下信息：
1.用户ID
2.用户设置的密码
3.用户的电子邮件地址
如果用户没有提供此信息，则需要提示用户提供该信息。
如果用户提供了这些信息，则需要查询数据库。如果用户ID和密码匹配，您需要通知用户验证码已发送到他们的电子邮件，需要进行验证。
如果用户没有新问题，您应该回复带有 "customer service" 的特殊令牌，以结束任务。
"""

class SmartAssistant:
    def __init__(self):
        self.client = client 

        self.system_prompt = sys_prompt
        self.registered_prompt = registered_prompt
        self.query_prompt = query_prompt
        self.delete_prompt = delete_prompt

        # Using a dictionary to store different sets of messages
        self.messages = {
            "system": [{"role": "system", "content": self.system_prompt}],
            "registered": [{"role": "system", "content": self.registered_prompt}],
            "query": [{"role": "system", "content": self.query_prompt}],
            "delete": [{"role": "system", "content": self.delete_prompt}]
        }

        # Current assignment for handling messages
        self.current_assignment = "system"

    def get_response(self, user_input):
        self.messages[self.current_assignment].append({"role": "user", "content": user_input})
        while True:
            response = self.client.chat.completions.create(
                model=chat_model,
                messages=self.messages[self.current_assignment],
                temperature=0.9,
                stream=False,
                max_tokens=2000,
            )

            ai_response = response.choices[0].message.content
            if "registered workers" in ai_response:
                self.current_assignment = "registered"
                print("意图识别:",ai_response)
                print("switch to <registered>")
                self.messages[self.current_assignment].append({"role": "user", "content": user_input})
            elif "query workers" in ai_response:
                self.current_assignment = "query"
                print("意图识别:",ai_response)
                print("switch to <query>")
                self.messages[self.current_assignment].append({"role": "user", "content": user_input})
            elif "delete workers" in ai_response:
                self.current_assignment = "delete"
                print("意图识别:",ai_response)
                print("switch to <delete>")
                self.messages[self.current_assignment].append({"role": "user", "content": user_input})
            elif "customer service" in ai_response:
                print("意图识别:",ai_response)
                print("switch to <customer service>")
                self.messages["system"] += self.messages[self.current_assignment]
                self.current_assignment = "system"
                return ai_response
            else:
                self.messages[self.current_assignment].append({"role": "assistant", "content": ai_response})
                return ai_response

    def start_conversation(self):
        while True:
            user_input = input("User: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting conversation.")
                break
            response = self.get_response(user_input)
            print("Assistant:", response)

if __name__ == "__main__":
    assistant = SmartAssistant()
    assistant.start_conversation()