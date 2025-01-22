from vertexai.generative_models import GenerativeModel 
from tools.serp import search as google_search
from tools.wiki import search as wiki_search
from vertexai.generative_models import Part 
from utils.io import write_to_file
from config.logging import logger
from config.setup import config
from llm_openai import generate
from utils.io import read_file
from pydantic import BaseModel
from typing import Callable
from pydantic import Field 
from typing import Union
from typing import List 
from typing import Dict 
from enum import Enum
from enum import auto
import json
import os


Observation = Union[str, Exception]

INPUT_DIR = "./data/input/"
OUTPUT_DIR = "./data/output/"

class Name(Enum):
    WIKIPEDIA = auto()
    GOOGLE = auto()
    NONE = auto()

    def __str__(self) -> str:
        return self.name.lower()


class Choice(BaseModel):
    name: Name = Field(..., description="The name of the tool chosen.")
    reason: str = Field(..., description="The reason for choosing this tool.")


class Message(BaseModel):
    role: str = Field(..., description="The role of the message sender.")
    content: str = Field(..., description="The content of the message.")


class Tool:
   
    def __init__(self, name: Name, func: Callable[[str], str]):
        self.name = name
        self.func = func

    def use(self, query: str) -> Observation:
        try:
            return self.func(query)
        except Exception as e:
            logger.error(f"Error executing tool {self.name}: {e}")
            return str(e)


class Agent:
    
    def __init__(self,  prompt_template_path: str, output_trace_path: str) -> None:
        
        self.prompt_template_path = prompt_template_path 
        self.output_trace_path = output_trace_path 
        self.tools: Dict[Name, Tool] = {}
        self.messages: List[Message] = []
        self.query = ""
        self.max_iterations = 5
        self.current_iteration = 0
        self.template = self.load_template()

    def load_template(self) -> str:
        return read_file(self.prompt_template_path)

    def register(self, name: Name, func: Callable[[str], str]) -> None:
        self.tools[name] = Tool(name, func)

    def trace(self, role: str, content: str) -> None:
        if role != "system":
            self.messages.append(Message(role=role, content=content))
        write_to_file(path=self.output_trace_path, content=f"{role}: {content}\n")

    def get_history(self) -> str:
        return "\n".join([f"{message.role}: {message.content}" for message in self.messages])
    

    def think(self) -> None:
        self.current_iteration += 1
        logger.info(f"Starting iteration {self.current_iteration}")
        write_to_file(path=self.output_trace_path, content=f"\n{'='*50}\nIteration {self.current_iteration}\n{'='*50}\n")

        if self.current_iteration > self.max_iterations:
            logger.warning("Reached maximum iterations. Stopping.")
            self.trace("assistant", "I'm sorry, but I couldn't find a satisfactory answer within the allowed number of iterations. Here's what I know so far: " + self.get_history())
            return

        prompt = self.template.format(
            query=self.query, 
            history=self.get_history(),
            tools=', '.join([str(tool.name) for tool in self.tools.values()])
        )

        response = self.ask_openai(prompt)
        logger.info(f"Thinking => {response}")
        self.trace("assistant", f"Thought: {response}")
        self.decide(response)

    def decide(self, response: str) -> None:
        try:
            cleaned_response = response.strip().strip('`').strip()
            if cleaned_response.startswith('json'):
                cleaned_response = cleaned_response[4:].strip()
            
            parsed_response = json.loads(cleaned_response)
            
            if "action" in parsed_response:
                action = parsed_response["action"]
                tool_name = Name[action["name"].upper()]
                if tool_name == Name.NONE:
                    logger.info("No action needed. Proceeding to final answer.")
                    self.think()
                else:
                    self.trace("assistant", f"Action: Using {tool_name} tool")
                    self.act(tool_name, action.get("input", self.query))
            elif "answer" in parsed_response:
                self.trace("assistant", f"Final Answer: {parsed_response['answer']}")
            else:
                raise ValueError("Invalid response format")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response: {response}. Error: {str(e)}")
            self.trace("assistant", "I encountered an error in processing. Let me try again.")
            self.think()
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            self.trace("assistant", "I encountered an unexpected error. Let me try a different approach.")
            self.think()

    def act(self, tool_name: Name, query: str) -> None:
        tool = self.tools.get(tool_name)
        if tool:
            result = tool.use(query)
            observation = f"Observation from {tool_name}: {result}"
            self.trace("system", observation)
            self.messages.append(Message(role="system", content=observation))  # Add observation to message history
            self.think()
        else:
            logger.error(f"No tool registered for choice: {tool_name}")
            self.trace("system", f"Error: Tool {tool_name} not found")
            self.think()

    def execute(self, query: str) -> str:
        self.query = query
        self.trace(role="user", content=query)
        self.think()
        return self.messages[-1].content

    def ask_openai(self, prompt: str) -> str:
        response = generate(prompt)
        return response if response else "No response from OpenAI"

def run(query: str) -> str:
    
    agent = Agent()
    agent.register(Name.WIKIPEDIA, wiki_search)
    agent.register(Name.GOOGLE, google_search)
    return agent.execute(query)

def run_all_templates(query: str):
    for template_file in os.listdir(INPUT_DIR):
        if template_file.endswith(".txt"):
            template_path = os.path.join(INPUT_DIR, template_file)
            output_file = template_file.replace(".txt", "_trace.txt")
            output_path = os.path.join(OUTPUT_DIR, output_file)

            # Log current execution
            logger.info(f"Running template: {template_file}")
            logger.info(f"Output will be saved to: {output_file}")

            # Initialize Agent with template and output paths
            agent = Agent(prompt_template_path=template_path, output_trace_path=output_path)
            agent.register(Name.WIKIPEDIA, wiki_search)
            agent.register(Name.GOOGLE, google_search)

            # Execute the agent with the query
            final_answer = agent.execute(query)

            # Save the trace to the output file
            with open(output_path, "a") as f:
                f.write(f"\nFinal Answer: {final_answer}\n")

            logger.info(f"Execution completed for {template_file}.")

if __name__ == "__main__":
    query = "What is the age of the oldest tree in the country that has won the most FIFA World Cup titles?"
    run_all_templates(query)