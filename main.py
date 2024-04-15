from langchain_community.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import(
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)

from langchain.schema import SystemMessage
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

from tools.sql import run_query_tool, list_tables, describe_tables_tool
from tools.report import write_report_tool
from handlers.chat_model_start_handler import ChatModelStartHandler

load_dotenv()
handler = ChatModelStartHandler()
chat= ChatOpenAI(
    callbacks=[handler]
)

tables= list_tables()

prompt= ChatPromptTemplate(
    messages=[
        SystemMessage(content=f"You are an AI that has access to a SQLite Database."
                      f"The Database has the following tables: {tables}\n"
                      "Don't make any assumptions about data and only use the 'describe_tables' functions to know more about each table schema"),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
)

memory= ConversationBufferMemory(memory_key="chat_history", return_messages=True)

tools = [run_query_tool, describe_tables_tool, write_report_tool]

agent = create_openai_functions_agent(
    llm=chat,
    prompt= prompt,
    tools= tools
)

agent_executor= AgentExecutor(
    agent=agent,
    verbose=False,
    tools=tools,
    memory=memory,
)

agent_executor.invoke({"input":"list the highest products in price"})

#agent_executor.invoke({"input":"Repeat the exact same process for users"})