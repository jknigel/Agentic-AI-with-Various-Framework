# %% [markdown]
# <p style="text-align:center">
#     <a href="https://skills.network" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
#     </a>
# </p>
# 

# %% [markdown]
# # **Agents with Tools versus Tasks with Tools in CrewAI**
# 

# %% [markdown]
# Estimated time needed: **30** minutes
# 
# 

# %% [markdown]
# Build a specialized agentic AI chatbot with CrewAI in only 30 minutes! This lab will teach you how to use tools with agentic workflows to solve real-world challenges. You'll create and orchestrate AI agents, define multi-step workflows, and explore the differences between assigning tools at the Agent level versus the Task level. This hands-on project is a necessity for all software and machine learning engineers looking to build more efficient and reliable multi-agent systems.
# 
# Here's your challenge: You've been hired by "The Daily Dish," a popular restaurant with a customer service problem. Chef Maria, the owner, needs your help: "My team spends hours answering the same questions about reservations, menu details, and our location," she explains. "Build me a chatbot that can handle these inquiries intelligently, so my staff can focus on creating exceptional dining experiences." Chef Maria provides a PDF of frequently asked questions (FAQs) to serve as the primary knowledge base.
# 
# You'll solve this by building a crew of AI agents that can search the FAQ document, browse the web for supplementary information, and synthesize the findings into a friendly, helpful response for the customer. Most importantly, you will learn how to make this process more robust and efficient by assigning tools directly to the tasks that need them.
# 

# %% [markdown]
# 
# ## __Table of Contents__<a id="toc"></a>
# <ol>
#     <li><a href="#Learning-Objectives">Learning Objectives</a></li>
#     <li>
#         <a href="#Setup">Setup</a>
#         <ol>
#             <li><a href="#Installing-Required-Libraries">Installing Required Libraries</a></li>
#             <li><a href="#Importing-Required-Libraries">Importing Required Libraries</a></li>
#         </ol>
#     </li>
#     <li>
#         <a href="#Project-Roadmap">Project Roadmap</a>
#         <ol>
#             <li><a href="#Agentic-AI-with-CrewAI">Agentic AI with CrewAI</a></li>
#             <li><a href="#The-Role-of-Tools:-Agent-Centric-versus-Task-Centric">The Role of Tools: Agent-Centric versus Task-Centric</a></li>
#         </ol>
#     </li>
#     <li><a href="#Writing-the-Code">Writing the Code</a>
#         <ol>
#             <li><a href="#Configuring-our-LLM">Configuring our LLM</a></li>
#             <li><a href="#Defining-Tools-for-Information-Gathering">Defining Tools for Information Gathering</a></li>
#             <li><a href="#Approach-1:-The-Standard-Method-(Agent-Centric-Tools)">Approach 1: The Standard Method (Agent-Centric Tools)</a></li>
#             <li><a href="#Approach-2:-A-More-Focused-Method-(Task-Centric-Tools)">Approach 2: A More Focused Method (Task-Centric Tools)</a></li>
#             <li><a href="#Chatbot-Execution">Chatbot Execution</a></li>
#         </ol>
#     </li>
#     <li><a href="#Conclusion">Conclusion</a></li>
#     <li><a href="#Extending-CrewAI-with-Custom-Functions">Extending CrewAI with Custom Functions</a></li>
# </ol>
# 

# %% [markdown]
# ## Learning Objectives
# 
# After completing this lab you will be able to:
# 
# - Build a customer service chatbot using a multi-agent CrewAI workflow.
# - Implement tools for information retrieval from both local documents (PDFs) and the web.
# - Understand and contrast the two primary methods for tool usage in CrewAI: Agent-level versus Task-level assignment.
# - Design more efficient, predictable, and maintainable agentic workflows by assigning tools directly to tasks.
# 

# %% [markdown]
# ---
# 

# %% [markdown]
# ## Setup
# 

# %% [markdown]
# For this lab, we will be using the following libraries:
# 
# * [`crewai`](https://docs.crewai.com/introduction) for creating AI Agents and orchestrating workflows.
# * [`crewai_tools`](https://docs.crewai.com/core-concepts/Tools/) to equip our AI Agents with powerful pre-built tools.
# * [`langchain-community`](https://python.langchain.com/docs/integrations/tools/) for additional community-provided tools.
# * [`langchain-huggingface`](https://python.langchain.com/docs/integrations/providers/huggingface/) to power our PDF searching tool.
# * [`sentence-transformers`](https://python.langchain.com/docs/integrations/text_embedding/sentence_transformers/) as the embedding model for our PDF searching tool.
# 

# %% [markdown]
# ### Installing Required Libraries
# 
# Installing the libraries might take a few minutes. If you find any pip dependency errors, simply restart the kernel by going to `Kernel -> Restart Kernel` and execute the next cell (not the one below).
# 

# %%
#%%capture
#%pip install crewai==0.186.1
#%pip install crewai-tools==0.71.0
#%pip install langchain-community==0.3.29
#%pip install langchain-huggingface==0.3.1
#%pip install sentence-transformers==5.1.0

# %% [markdown]
# ### Importing Required Libraries
# 

# %%
#%%capture

from crewai import Agent, Task, Crew, Process
from crewai import LLM
from crewai_tools import PDFSearchTool, SerperDevTool

# %% [markdown]
# ---
# 

# %% [markdown]
# ## Project Roadmap
# 

# %% [markdown]
# ### Agentic AI with CrewAI
# 
# At a high level, instead of designing complex, rigid code for this chatbot, we delegate the work to a team—or **Crew**—of AI Agents.
# 
# An **AI Agent** is an autonomous worker. Like any worker, it needs a clear job description, which we provide through specific parameters:
# - **Role:** What is its primary function? (for example, "Customer Service Specialist")
# - **Goal:** What is it trying to achieve? (for example, "Answer customer questions accurately")
# - **Backstory:** What context does it need to perform its role effectively?
# - **Tools:** What resources can it use to accomplish its goal? (for example, a PDF search tool, a web browser)
# 
# Once we have our agents, we define the **Tasks** they need to complete. A **Crew** then orchestrates this entire process, managing the agents and ensuring tasks are executed in the correct order to achieve the final objective.
# 

# %% [markdown]
# ### The Role of Tools: Agent-Centric versus Task-Centric
# 
# A key decision in CrewAI is *how* to give tools to your agents. There are two main strategies, and understanding the difference is crucial for building robust applications.
# 
# **1. Agent-Centric Approach (The Generalist):**
# This is the most common method. You give an agent a 'toolbox' containing all the tools it might possibly need. The agent then uses its intelligence (the LLM's reasoning ability) to decide which tool is best for the situation at hand. This is flexible but can sometimes lead to the agent making mistakes or using tools inefficiently.
# 
# ![Diagram 1](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/NPYpo2eWUERulpzrm9xYtw/Adobe%20Express%20-%20file.jpg)
# 
# **2. Task-Centric Approach (The Specialist):**
# In this more advanced approach, you don't give the tools to the agent directly. Instead, you attach specific tools to the specific **Tasks** that require them. When the agent starts a task, it is temporarily granted access to only the tools needed for that job. This creates a much more focused and predictable workflow.
# 
# ![Diagram 2.jpg](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/olP9UG8s4B7QnOJhCRH4aA/Diagram%202.jpg)
# 

# %% [markdown]
# ---
# 

# %% [markdown]
# ## Writing the Code
# 

# %% [markdown]
# ### Configuring our LLM
# 
# First, we configure the Language Model (LLM) that will power our agents' reasoning and language generation. We're using IBM's WatsonX `Granite 3.3` model. Although other models could be used, this one provides a good balance of performance and cost for our use case.
# 

# %%
llm = LLM(
    model="watsonx/ibm/granite-3-3-8b-instruct",
    base_url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
)

# %% [markdown]
# ### Defining Tools for Information Gathering
# 
# Our chatbot needs to access information from two sources: Daily Dish's FAQ PDF and the general internet. To do this, we'll instantiate two tools.
# 
# 1.  **`PDFSearchTool`**: This tool, from the `crewai_tools` library, will index the provided PDF document. When used, it performs a semantic search to find the most relevant sections of the PDF related to a query.
# 2.  **`SerperDevTool`**: This simple tool from `crewai_tools` allows an agent to perform a web search using SerperAPI.
# 
# 
# To use `SerperDevTool`, we first need an **API key**—a crucial credential that grants access to web searches. Once we have the API key, we can import the library and integrate it into our customer service workflow.  
# 
# > You will need to get the API Key from `serper.dev` website. To get the API key, create an account using your email if you have registered or login if you already have an account. Once you are logged in, you will be on the `Dashboard` section. Click on `API Keys` and you will find your API key. Copy the key and replace it with `API_KEY` here.
# 
# To get the API key and for more details on `SerperDevTool` and its capabilities, visit the official webpage: [SerperDevTool Documentation](https://serper.dev/).
# 

# %% [markdown]
# ### Initializing Serper Dev Tool: 
# 

# %%
import os
os.environ['SERPER_API_KEY'] = 'API_KEY' 

# %%
web_search_tool = SerperDevTool()

# %% [markdown]
# ### Creating our PDF Search Tool: 
# 

# %%
import warnings
warnings.filterwarnings('ignore') #Keeps Jupyter Notebook clean (not part of functionality)

pdf_search_tool = PDFSearchTool(
    pdf="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/7vgNfis17dQfjHAiIKkBOg/The-Daily-Dish-FAQ.pdf",
    config=dict(
        embedder=dict(
            provider="huggingface",
            config=dict(
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
        )
    )
)

# %% [markdown]
# ### Approach 1: The Standard Method (Agent-Centric Tools)
# 
# First, we'll build the chatbot using the conventional approach where we give our agent a toolbox with all the necessary tools. The agent will be responsible for deciding whether to search the PDF or generate a response with just it's LLM.
# 
# #### **Step 1.1: Create the Agent**
# 
# We define an `Inquiry Specialist Agent` whose job is to answer questions. Notice that the `tools` parameter is a list containing the `pdf_search_tool` and `web_search_tool`.
# 

# %%
agent_centric_agent = Agent(
    role="The Daily Dish Inquiry Specialist",
    goal="""Accurately answer customer questions about The Daily Dish restaurant. 
    You must decide whether to use the restaurant's FAQ PDF or a web search to find the best answer.""",
    backstory="""You are an AI assistant for 'The Daily Dish'.
    You have access to two tools: one for searching the restaurant's FAQ document and another for searching the web.
    Your job is to analyze the user's question and choose the most appropriate tool to find the information needed to provide a helpful response.""",
    tools=[pdf_search_tool, web_search_tool],
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# %% [markdown]
# #### **Step 1.2: Define the Task**
# 
# We create a single, broad task that instructs the agent to handle the customer's query.
# 

# %%
agent_centric_task = Task(
    description="Answer the following customer query: '{customer_query}'. "
                "Analyze the question and use the tools at your disposal (PDF search or web search) to find the most relevant information. "
                "Synthesize the findings into a clear and friendly response.",
    expected_output="A comprehensive and well-formatted answer to the customer's query.",
    agent=agent_centric_agent
)

# %% [markdown]
# #### **Step 1.3: Assemble the Crew**
# 
# Finally, we create the Crew. It's a simple setup with our one agent and one task.
# 

# %%
agent_centric_crew = Crew(
    agents=[agent_centric_agent],
    tasks=[agent_centric_task],
    process=Process.sequential,
    verbose=False
)

# %% [markdown]
# Before executing the crew, let's download the FAQ file that we have curated with all the questions and answers. As you execute the below line of code, the file will be downloaded, which you will be able to access from the file library on the left of your screen.
# 

# %%
# Download the FAQ document for the tool to use
#!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/7vgNfis17dQfjHAiIKkBOg/The-Daily-Dish-FAQ.pdf

# %% [markdown]
# Try asking the following questions:
# 
# 1. What are the timings?
# 2. What is the phone number?
# 3. What is the location?
# 
# You could also ask some combinations of other questions that you will find in the FAQ PDF itself.
# 

# %%
print("\nWelcome to The Daily Dish Chatbot!")
print("What would you like to know? (Type 'exit' to quit)")

while True: 
    user_input = input("\nYour question: ").lower()
    if user_input == 'exit':
        print("Thank you for chatting. Have a great day!")
        break
    
    if not user_input:
        print("Please type a question.")
        continue

    try:
        # Here we use our more advanced, task-centric crew
        result_agent_centric = agent_centric_crew.kickoff(inputs={'customer_query': user_input})
        print("\n--- The Daily Dish Assistant ---")
        print(result_agent_centric)
        print("--------------------------------")
    except Exception as e:
        print(f"An error occurred: {e}")

# %% [markdown]
# This approach works, and because we set `verbose=True` to Agent, we see the agent spending thought process on *which tool to choose*. For a simple query, this is fine. But in a complex workflow with many tools and steps, this ambiguity can lead to errors or inefficient tool usage. Now, let's see a better way.
# 

# %% [markdown]
# ### Approach 2: A More Focused Method (Task-Centric Tools)
# 
# Now, we'll refactor our solution to use a task-centric approach. We will create a multi-step process where each step (Task) has its own dedicated tool. This makes the agent's job simpler and the overall workflow more reliable.
# 
# Our new workflow will have two tasks:
# 1.  **Search the FAQ:** This task will *only* use the `PDFSearchTool`.
# 2.  **Draft the Response:** This task will use the information from the first two tasks to write the final answer. It needs no tools.
# 
# #### **Step 2.1: Create the Agent**
# 
# This time, we create a `Customer Service Specialist` agent. Notice the critical difference: the `tools` list is **empty**. We are not giving the agent its toolbox upfront. However, even if we gave the agent many many tools, they would simply be completely overriden by the tools in the task.
# 

# %%
task_centric_agent = Agent(
    role="Customer Service Specialist",
    goal="Provide exceptional customer service by following a multi-step process to answer customer questions accurately.",
    backstory="""You are an AI assistant for 'The Daily Dish'.
    You are an expert at following instructions. You will be given a sequence of tasks to complete.
    For each task, you will be provided with the specific tool needed to accomplish it.
    Your job is to execute each task diligently and pass the results to the next step.""",
    tools=[], # The agent is not given any tools directly
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# %% [markdown]
# #### **Step 2.2: Define the Tasks with Specific Tools**
# 
# Here is the core of the new approach. We define two distinct tasks. For the first two, we use the `tools` parameter within the `Task` definition itself to assign a specific tool to that step.
# 
# - **`faq_search_task`**: Is exclusively paired with `pdf_search_tool`.
# - **`response_drafting_task`**: Needs the output (context) from the previous tasks but requires no tools of its own.
# 

# %%
faq_search_task = Task(
    description="Search the restaurant's FAQ PDF for information related to the customer's query: '{customer_query}'.",
    expected_output="A snippet of the most relevant information from the PDF, or a statement that the information was not found.",
    tools=[pdf_search_tool], # Tool assigned directly to the task
    agent=task_centric_agent
)

response_drafting_task = Task(
    description="Using the information gathered from the FAQ search, draft a friendly and comprehensive response to the customer's query: '{customer_query}'.",
    expected_output="The final, customer-facing response.",
    agent=task_centric_agent,
    context=[faq_search_task]
)

# %% [markdown]
# #### **Step 2.3: Assemble the New Crew**
# 
# We assemble our new crew, providing the single agent and the list of three tasks. The `Process.sequential` setting ensures the tasks run in the order we've listed them.
# 

# %%
task_centric_crew = Crew(
    agents=[task_centric_agent],
    tasks=[faq_search_task, response_drafting_task],
    process=Process.sequential,
    verbose=True
)

# %% [markdown]
# ### Chatbot Execution
# 
# This final script runs the interactive chatbot. It uses a `while` loop to repeatedly get user input. The core of the interaction is `task_centric_crew.kickoff(...)`. This method activates the entire task-centric system we just defined.
# 
# When you run this, the agent will no longer have to *think* about which tool to use. It will simply execute Task 1 with the PDF tool, then Task 2 with the web search tool, and finally Task 3 to generate the response. The process is deterministic, efficient, and easier to debug.
# 

# %% [markdown]
# Try asking the following questions:
# 
# 1. What are the timings?
# 2. What is the phone number?
# 3. What is the location?
# 
# You could also ask some combinations of other questions that you will find in the FAQ PDF itself.
# 

# %%
print("\nWelcome to The Daily Dish Chatbot!")
print("What would you like to know? (Type 'exit' to quit)")

while True: 
    user_input = input("\nYour question: ").lower()
    if user_input == 'exit':
        print("Thank you for chatting. Have a great day!")
        break
    
    if not user_input:
        print("Please type a question.")
        continue

    try:
        # Here we use our more advanced, task-centric crew
        result_task_centric = task_centric_crew.kickoff(inputs={'customer_query': user_input})
        print("\n--- The Daily Dish Assistant ---")
        print(result_task_centric)
        print("--------------------------------")
    except Exception as e:
        print(f"An error occurred: {e}")

# %% [markdown]
# <!-- ## Conclusion
# 
# In this lab, you built a sophisticated customer service chatbot and, more importantly, explored a fundamental concept in CrewAI: the strategic assignment of tools.
# 
# You saw two approaches:
# 1.  **Agent-Centric:** Flexible and easy to set up, but relies on the agent's reasoning to select tools, which can be inefficient or unpredictable in complex scenarios.
# 2.  **Task-Centric:** More structured and robust. By assigning tools directly to the tasks that need them, you create a clear, deterministic, and efficient workflow.
# 
# **Key Advantages of Task-Centric Tools:**
# - **Focus & Efficiency:** The agent doesn't waste time or processing power deciding which tool to use. It's told exactly what to use for each step.
# - **Clarity & Maintainability:** The workflow is explicit and easy to follow. Anyone reading the code can see precisely which task uses which tool.
# - **Control & Security:** This pattern allows for fine-grained control, granting an agent access to a powerful tool only for the specific duration of a single task.
# 
# Although the agent-centric approach has its place, mastering the task-centric method is a key step toward building truly professional, production-grade multi-agent systems with CrewAI.
#  -->
# 
# ## Conclusion
# 
# In this lab, you built a sophisticated customer service chatbot and, more importantly, explored a fundamental concept in CrewAI: the strategic assignment of tools.
# 
# You saw two approaches:
# 1. **Agent‑Centric:** Flexible and easy to set up, but relies on the agent's reasoning to select tools, which can be inefficient or unpredictable in complex scenarios.  
# 2. **Task‑Centric:** More structured and robust. By assigning tools directly to the tasks that need them, you create a clear, deterministic, and efficient workflow.
# 
# | Aspect             | Agent‑Centric Output                                                   | Task‑Centric Output                                                       |
# |--------------------|-------------------------------------------------------------------------|----------------------------------------------------------------------------|
# | **Predictability** | Varies run‑to‑run based on the LLM’s tool choice and phrasing.          | Consistent across runs thanks to fixed task‑to‑tool mapping.              |
# | **Debuggability**  | Harder to trace—calls and errors are buried in the LLM’s reasoning.     | Easy to debug—each task’s inputs, outputs, and errors are explicitly logged. |
# | **Reusability**    | You get a free‑form text blob that you must parse yourself.             | You get structured intermediate results (for example, JSON or code blocks) ready for reuse. |
# | **Structure**¹     | Search and formatting are blended in one step.                          | A separate formatting task produces a clean, structured final message.    |
# 
# This “Structure” difference exists only because we introduced a dedicated formatting task in the Task‑Centric workflow.
# 
# **Key Advantages of Task‑Centric Tools:**
# - **Focus & Efficiency:** The agent doesn't waste time or processing power deciding which tool to use. It's told exactly what to use for each step.  
# - **Clarity & Maintainability:** The workflow is explicit and easy to follow. Anyone reading the code can see precisely which task uses which tool.  
# - **Control & Security:** This pattern allows for fine‑grained control, granting an agent access to a powerful tool only for the specific duration of a single task.
# 
# While the agent‑centric approach has its place, mastering the task‑centric method is a key step toward building truly professional, production‑grade multi‑agent systems with CrewAI.
# 

# %% [markdown]
# ## Extending CrewAI with Custom Functions
# 

# %% [markdown]
# In CrewAI, **tools** (or Functions) are functional components that agents can call to perform specific actions, such as searching the web, reading PDFs, or doing calculations. Although CrewAI provides several built-in tools, we can also create our **own custom tools** to extend its capabilities for domain-specific or utility tasks.
# 
# Custom tools are simply Python functions that are **wrapped using the `@tool` decorator** from `crewai.tools`. This allows CrewAI agents to recognize, reason about, and invoke those functions during task execution.
# 

# %% [markdown]
# Let's see how we can create an **add tool** using CrewAI’s `@tool` decorator. Just like LangChain allows custom tool creation using its own `@tool` decorator, CrewAI also provides a simple way to register functions as tools that agents can invoke during task execution.
# 
# To define a tool in CrewAI:
# 
# - You annotate your function with `@tool("Tool Name")`.
# - You provide a **docstring**, which acts as the tool's self-description and helps the language model understand what the tool does.
# - Then you implement the **function body**, which contains the actual logic.
# 

# %%
from crewai.tools import tool
import re

@tool("Add Two Numbers Tool")
def add_numbers(data: str) -> int:
    """
    Extracts and adds integers from the input string.
    Example input: 'add 1 and 2' or '[1,2,3,4]'
    Output: sum of the numbers
    """
    # Find all integers in the string
    numbers = list(map(int, re.findall(r'-?\d+', data)))
    return sum(numbers)

# %% [markdown]
# Now, let's create another tool called `multiply_numbers`. This tool takes a string input such as `"multiply 2 and 3"` or `"values are [2,3,4]"`, extracts all the integers from the text, and returns their **product** as an integer.
# 

# %%
from functools import reduce

@tool("Multiply Numbers Tool")
def multiply_numbers(data: str) -> int:
    """
    Extracts and multiplies integers from the input string.
    Example input: 'multiply 2 and 3' or '[2,3,4]'
    Output: the product of all numbers found
    """
    numbers = list(map(int, re.findall(r'-?\d+', data)))
    return reduce(lambda x, y: x * y, numbers, 1)

# %% [markdown]
# Now, we create a `Calculator Agent` with a clear goal being to extract, add or multiply numbers based on a user's query. We provide the relevant backstory, the tools that we created, and the LLM itself.
# 

# %%
calculator_agent = Agent(
    role="Calculator",
    goal="Extracts, adds, or multiplies numbers when asked, using the Add Two Numbers and Multiply Numbers tools.",
    backstory="An expert at parsing numeric instructions and computing sums or products.",
    tools=[add_numbers, multiply_numbers],
    llm=llm,
    allow_delegation=False
)

# %% [markdown]
# We also create a `Calculation Task` by providing a clear description, an expected output, and an agent.
# 

# %%
calculation_task = Task(
    description="Extract numbers from '{numbers}' and either add or multiply them, depending on the natural-language instruction.",
    expected_output="An integer result (sum or product) based on the user’s request.",
    agent=calculator_agent
)

# %% [markdown]
# Now let's bring together the created agent and task in a `Crew`.
# 

# %%
crew = Crew(
    agents=[calculator_agent],
    tasks=[calculation_task],
    # verbose=True #Uncomment this to see the steps taken to get the final answer
)

# %% [markdown]
# Let's run the crew by providing a user query in `.kickoff()` and check the output.
# 

# %%
# Inputs for addition…
result = crew.kickoff(inputs={'numbers': 'please add 4, 5, and 6'})
print("Sum result:", result)

# %%
# Inputs for multiplication…
result = crew.kickoff(inputs={'numbers': 'multiply 7 and 8 also 9 dont forget 10'})
print("Product result:", result)

# %% [markdown]
# ## Authors
# 
# [Abdul Fatir](https://www.linkedin.com/in/abdul-fatir)
# 
# Abdul specializes in Data Science, Machine Learning, and AI. He has deep expertise in understanding how the latest technologies work, and their applications.  
# Feel free to contact him with questions about this project or any other AI/ML topics.
# 
# [Karan Goswami](https://author.skills.network/instructors/karan_goswami) is a Data Scientist at IBM and is pursuing Master's student at McMaster university with a major in AI. He is fluent in Generative AI, AI/ML topics. He has many projects published on [CognitiveClass.ai](https://cognitiveclass.ai). Feel free to connect with him if you need any help or just want to connect!
# 

# %% [markdown]
# ## Change Log
# 
# <details>
#     <summary>Click here for the changelog</summary>
# 
# |Date (YYYY-MM-DD)|Version|Changed By|Change Description|
# |-|-|-|-|
# |2025-07-27|1.0|Abdul Fatir|Initial version created|
# |2025-08-02|1.1|Steve Ryan|ID review and format fixes|
# |2025-08-04|1.2|Leah Hanson|QA review and grammar/IBM style guide adherencefixes|
# </details>
# 
# ---
# 
# 

# %% [markdown]
# Copyright © IBM Corporation. All rights reserved.
# 

# %%



