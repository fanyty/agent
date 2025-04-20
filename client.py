import os
from dotenv import load_dotenv
from openai import OpenAI
import re # Need re for parsing

# --- Environment Setup ---
# Load environment variables from .env file (especially API keys and endpoints)
load_dotenv()

# Read API configuration from environment variables
api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('BASE_URL') # Potentially a non-OpenAI endpoint
chat_model = os.getenv('CHAT_MODEL') # The specific LLM to use

# Initialize the OpenAI client with the specified API key and base URL
# This client will be used for all interactions with the LLM
client = OpenAI(
    api_key = api_key,
    base_url = base_url
)

# --- Prompt Engineering ---
# Define the system prompts that guide the LLM's behavior for different tasks.

# The main system prompt: defines the agent's overall role as a customer service dispatcher
# and instructs it to use specific tokens to switch between different business processes.
sys_prompt = """你是一个聪明的客服。您将能够根据用户的问题将不同的任务分配给不同的人。您有以下业务线：
1.用户注册。如果用户想要执行这样的操作，您应该发送一个带有"registered workers"的特殊令牌。并告诉用户您正在调用它。
2.用户数据查询。如果用户想要执行这样的操作，您应该发送一个带有"query workers"的特殊令牌。并告诉用户您正在调用它。
3.删除用户数据。如果用户想执行这种类型的操作，您应该发送一个带有"delete workers"的特殊令牌。并告诉用户您正在调用它。
"""

# Prompt for the registration process: instructs the LLM on what information to collect,
# how to confirm it, and the specific format to output the collected data in before finishing.
# The format "用户信息：[...]" is crucial for later parsing.
registered_prompt = """
您的任务是根据用户信息存储数据。您需要从用户那里获得以下信息：
1.用户名、性别、年龄 (例如：姓名: 张三, 性别: 男, 年龄: 25)
2.用户设置的密码 (例如：密码: password123)
3.用户的电子邮件地址 (例如：邮箱: zhangsan@example.com)
如果用户没有提供此信息，您需要提示用户提供。收集完所有信息后，请确认信息并准备存储。
最后，请明确告知用户注册成功，并按照以下格式包含用户信息：
'注册成功！用户信息：[姓名: <姓名>, 性别: <性别>, 年龄: <年龄>, 密码: <密码>, 邮箱: <邮箱>]'
然后回复带有 "customer service" 的特殊令牌，以结束任务。
"""

# Prompt for the query process: instructs the LLM on what information to collect (ID, password)
# and the format to output the query criteria in before finishing.
# The format "查询条件：[...]" is crucial for later parsing.
query_prompt = """
您的任务是查询用户信息。您需要从用户那里获得以下信息：
1.用户ID (例如：用户ID: 10001)
2.用户设置的密码 (例如：密码: password123)
如果用户没有提供此信息，则需要提示用户提供。收集完信息后，请明确告知用户您将要查询，并按照以下格式包含查询条件：
'正在查询... 查询条件：[用户ID: <ID>, 密码: <密码>]'
然后回复带有 "customer service" 的特殊令牌，以结束任务。（实际查询将在后台完成）
"""

# Prompt for the delete process: instructs the LLM on what info to collect (ID, password, email)
# and the format to output the deletion criteria in before finishing.
# The format "删除条件：[...]" is crucial for later parsing.
delete_prompt = """
您的任务是删除用户信息。您需要从用户那里获得以下信息：
1.用户ID (例如：用户ID: 10001)
2.用户设置的密码 (例如：密码: password123)
3.用户的电子邮件地址 (例如：邮箱: zhangsan@example.com)
如果用户没有提供此信息，则需要提示用户提供该信息。收集完所有信息后，请确认信息并准备删除。
最后，请明确告知用户将进行删除操作（并模拟发送验证码），并按照以下格式包含条件：
'将删除用户... 删除条件：[用户ID: <ID>, 密码: <密码>, 邮箱: <邮箱>]'
然后回复带有 "customer service" 的特殊令牌，以结束任务。（实际删除将在后台完成）
"""

# --- Agent Class Definition ---
class SmartAssistant:
    """
    A conversational agent that handles customer service tasks like registration,
    querying, and deleting user information using an LLM and a state machine.

    Attributes:
        client: An OpenAI API client instance.
        system_prompt, registered_prompt, query_prompt, delete_prompt: Strings
            containing instructions for the LLM for different states.
        messages (dict): Stores separate conversation histories for each state
            ("system", "registered", "query", "delete").
        current_assignment (str): The current state of the agent, determining
            which prompt and message history to use.
        user_database (dict): An in-memory dictionary acting as a simulated database
            for user information. Data is lost on restart.
        next_user_id (int): A counter to generate unique user IDs for registration.
    """
    def __init__(self):
        """Initializes the SmartAssistant with API client, prompts, message histories,
           initial state, and the simulated database."""
        self.client = client

        # Store the prompts for easy access
        self.system_prompt = sys_prompt
        self.registered_prompt = registered_prompt
        self.query_prompt = query_prompt
        self.delete_prompt = delete_prompt

        # Initialize message history for each state. Each history starts
        # with the corresponding system prompt.
        self.messages = {
            "system": [{"role": "system", "content": self.system_prompt}],
            "registered": [{"role": "system", "content": self.registered_prompt}],
            "query": [{"role": "system", "content": self.query_prompt}],
            "delete": [{"role": "system", "content": self.delete_prompt}]
        }

        # Start in the main "system" state
        self.current_assignment = "system"

        # Initialize the in-memory database and user ID counter
        self.user_database = {}
        self.next_user_id = 10001

    # --- Helper methods for parsing LLM output ---
    # These methods use regular expressions to extract structured data
    # from the LLM's responses when a task is about to finish.

    def _parse_registration_info(self, text):
        """Parses registration details (name, gender, age, password, email)
           from the LLM response text based on the expected format.
           Returns a dictionary with the info or None if parsing fails."""
        info = {}
        # Regex patterns to find key-value pairs within the specific format
        patterns = {
            "name": r"姓名:\s*(\S+)",
            "gender": r"性别:\s*(\S+)",
            "age": r"年龄:\s*(\d+)",
            "password": r"密码:\s*(\S+)",
            "email": r"邮箱:\s*(\S+@\S+\.\S+)"
        }
        # Look for the specific marker "用户信息：[...]"
        match = re.search(r"用户信息：\[(.*?)\]", text)
        if not match:
            print(f"[Parsing Error] Registration marker not found in: {text}")
            return None
        data_str = match.group(1) # Extract the content within brackets

        # Extract each piece of information using regex
        for key, pattern in patterns.items():
            m = re.search(pattern, data_str)
            if m:
                info[key] = m.group(1)
            else:
                # If any piece is missing, parsing fails
                print(f"[Parsing Error] Missing '{key}' in registration info: {data_str}")
                return None

        # Convert age to integer, handle potential errors
        if 'age' in info:
            try:
                info['age'] = int(info['age'])
            except ValueError:
                 print(f"[Parsing Error] Invalid age format: {info['age']}")
                 return None
        return info

    def _parse_query_info(self, text):
        """Parses query details (user_id, password) from the LLM response.
           Returns a dictionary or None if parsing fails."""
        info = {}
        patterns = {
            "user_id": r"用户ID:\s*(\d+)",
            "password": r"密码:\s*(\S+)",
        }
        # Look for the marker "查询条件：[...]"
        match = re.search(r"查询条件：\[(.*?)\]", text)
        if not match:
            print(f"[Parsing Error] Query marker not found in: {text}")
            return None
        data_str = match.group(1)

        for key, pattern in patterns.items():
            m = re.search(pattern, data_str)
            if m:
                info[key] = m.group(1)
            else:
                print(f"[Parsing Error] Missing '{key}' in query info: {data_str}")
                return None

        # Convert user_id to integer
        if 'user_id' in info:
             try:
                info['user_id'] = int(info['user_id'])
             except ValueError:
                 print(f"[Parsing Error] Invalid user_id format: {info['user_id']}")
                 return None
        return info

    def _parse_delete_info(self, text):
        """Parses deletion details (user_id, password, email) from the LLM response.
           Returns a dictionary or None if parsing fails."""
        info = {}
        patterns = {
            "user_id": r"用户ID:\s*(\d+)",
            "password": r"密码:\s*(\S+)",
            "email": r"邮箱:\s*(\S+@\S+\.\S+)"
        }
        # Look for the marker "删除条件：[...]"
        match = re.search(r"删除条件：\[(.*?)\]", text)
        if not match:
            print(f"[Parsing Error] Delete marker not found in: {text}")
            return None
        data_str = match.group(1)

        for key, pattern in patterns.items():
            m = re.search(pattern, data_str)
            if m:
                info[key] = m.group(1)
            else:
                print(f"[Parsing Error] Missing '{key}' in delete info: {data_str}")
                return None

        # Convert user_id to integer
        if 'user_id' in info:
             try:
                info['user_id'] = int(info['user_id'])
             except ValueError:
                 print(f"[Parsing Error] Invalid user_id format: {info['user_id']}")
                 return None
        return info

    # --- Helper methods for simulated database operations ---
    # These methods interact with the in-memory `self.user_database` dictionary.

    def _register_user(self, info):
        """Adds a new user to the simulated database.
           Returns the newly generated user ID."""
        user_id = self.next_user_id
        self.user_database[user_id] = {
            "name": info["name"],
            "gender": info["gender"],
            "age": info["age"],
            "password": info["password"], # WARNING: In a real app, hash the password!
            "email": info["email"]
        }
        self.next_user_id += 1 # Increment for the next user
        print(f"[DB Action] Registered User ID: {user_id}, Info: {self.user_database[user_id]}")
        return user_id

    def _query_user(self, user_id, password):
        """Queries the simulated database for a user by ID and password.
           Returns user info (excluding password) if found and password matches,
           otherwise returns None."""
        user_info = self.user_database.get(user_id)
        # Check if user exists and password matches
        if user_info and user_info["password"] == password:
            print(f"[DB Action] Query Success for User ID: {user_id}")
            # Return user info, but exclude the password for security
            return {k: v for k, v in user_info.items() if k != 'password'}
        else:
            print(f"[DB Action] Query Failed for User ID: {user_id}")
            return None

    def _delete_user(self, user_id, password, email):
        """Deletes a user from the simulated database if ID, password, and email match.
           Returns True if deletion was successful, False otherwise."""
        user_info = self.user_database.get(user_id)
        # Check if user exists and all credentials match
        if user_info and user_info["password"] == password and user_info["email"] == email:
            del self.user_database[user_id]
            print(f"[DB Action] Deleted User ID: {user_id}")
            return True
        else:
            print(f"[DB Action] Delete Failed for User ID: {user_id}")
            return False

    # --- Core Agent Logic ---
    def get_response(self, user_input):
        """Processes user input, interacts with the LLM, manages state transitions,
           performs simulated DB operations, and returns the agent's response.

        Args:
            user_input (str): The text input from the user.

        Returns:
            str: The agent's response to the user.
        """
        # Get the message history for the current state
        current_messages = self.messages[self.current_assignment]
        # Add the latest user input to the history
        current_messages.append({"role": "user", "content": user_input})

        # Loop allows for potential internal state changes without returning immediately
        while True:
            # Call the LLM API
            response = self.client.chat.completions.create(
                model=chat_model,
                messages=current_messages, # Use history for the current state
                temperature=0.7, # Lower temperature for more deterministic behavior needed for parsing
                stream=False,
                max_tokens=2000,
            )

            # Extract the LLM's response text
            ai_response = response.choices[0].message.content

            # --- State Transition & Action Logic --- #

            # Check if the response indicates a switch to a specific worker (business process)
            next_assignment = None
            if "registered workers" in ai_response:
                next_assignment = "registered"
            elif "query workers" in ai_response:
                next_assignment = "query"
            elif "delete workers" in ai_response:
                next_assignment = "delete"

            # Check if the response indicates returning to the main customer service state
            elif "customer service" in ai_response:
                # This means the current task (e.g., registration) is finishing.
                # We need to parse the final response and perform the DB action.
                db_action_result_msg = "" # Additional message based on DB result
                previous_assignment = self.current_assignment # Remember which task finished

                # Perform action based on the completed task
                if previous_assignment == "registered":
                    parsed_info = self._parse_registration_info(ai_response)
                    if parsed_info:
                        new_user_id = self._register_user(parsed_info)
                        # Add the new user ID to the confirmation message
                        db_action_result_msg = f" (您的用户 ID 是: {new_user_id})"
                    else:
                        db_action_result_msg = " (错误：注册信息解析失败，未能存储用户)"
                elif previous_assignment == "query":
                    parsed_info = self._parse_query_info(ai_response)
                    if parsed_info:
                        user_data = self._query_user(parsed_info["user_id"], parsed_info["password"])
                        if user_data:
                            # Append the found user data to the message
                            db_action_result_msg = f" \n查询成功！您的信息如下： {user_data}"
                        else:
                            db_action_result_msg = " \n查询失败：用户ID或密码错误。"
                    else:
                        db_action_result_msg = " (错误：查询信息解析失败)"
                elif previous_assignment == "delete":
                     parsed_info = self._parse_delete_info(ai_response)
                     if parsed_info:
                         deleted = self._delete_user(parsed_info["user_id"], parsed_info["password"], parsed_info["email"])
                         if deleted:
                             db_action_result_msg = " \n用户删除成功！"
                         else:
                             db_action_result_msg = " \n删除失败：用户信息不匹配。"
                     else:
                         db_action_result_msg = " (错误：删除信息解析失败)"

                print(f"[State Transition] Task '{previous_assignment}' finished. Returning to 'system'.")
                # Append the final AI response (potentially modified with DB result) to the finished task's history
                current_messages.append({"role": "assistant", "content": ai_response + db_action_result_msg})
                # Merge the completed task's conversation history (excluding the initial system prompt)
                # into the main system history to maintain context for future interactions.
                self.messages["system"].extend(current_messages[1:])
                # Optional: Reset the task-specific history to save memory, keeping only the system prompt
                self.messages[previous_assignment] = [{"role": "system", "content": getattr(self, f"{previous_assignment}_prompt")}]
                # Switch state back to system
                self.current_assignment = "system"
                # Return the final combined response to the user
                return ai_response + db_action_result_msg

            # --- Handle State Transitions TO Workers ---
            if next_assignment:
                print(f"[State Transition] Detected trigger for '{next_assignment}'. Switching state.")
                # 1. Append the initial AI response that triggered the switch to the current history
                current_messages.append({"role": "assistant", "content": ai_response})
                # 2. Switch the current assignment state
                self.current_assignment = next_assignment
                # 3. Get the message list for the NEW state
                current_messages = self.messages[self.current_assignment]
                # 4. Add the *original* user input to the NEW state's history
                #    so the worker LLM knows what the user initially asked for.
                if not current_messages or current_messages[-1]["role"] != "user":
                     current_messages.append({"role": "user", "content": user_input})
                # 5. Continue the loop: The next iteration will call the LLM
                #    using the new state's prompt and history.
                continue

            # --- Standard Assistant Response (No State Change) ---
            else:
                # If no state change token was detected, it's a regular turn in the conversation.
                # Append the AI response to the current message history.
                current_messages.append({"role": "assistant", "content": ai_response})
                # Return the response to the user.
                return ai_response

    # --- Conversation Loop ---
    def start_conversation(self):
        """Starts the interactive command-line conversation loop with the user."""
        print("智能客服系统启动。输入 'exit' 或 'quit' 退出。")
        while True:
            # Get input from the user
            user_input = input("User: ")
            # Check for exit command
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting conversation.")
                break
            try:
                # Get the agent's response for the user input
                response = self.get_response(user_input)
                # Print the response
                print("Assistant:", response)
            except Exception as e:
                # Basic error handling for unexpected issues during API call or processing
                print(f"An error occurred: {e}")
                print("Assistant: 对不起，系统遇到了一些问题。请稍后再试或尝试重新开始对话。")
                # Consider resetting state or logging the error in a real application
                # self.current_assignment = "system"

# --- Main Execution Block ---
if __name__ == "__main__":
    # Create an instance of the agent
    assistant = SmartAssistant()
    # Start the conversation loop
    assistant.start_conversation()