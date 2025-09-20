# %% [markdown]
# <p style="text-align:center">
#     <a href="https://skills.network" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
#     </a>
# </p>
# 

# %% [markdown]
# # **CrewAI 101: Building Multi-Agent AI Systems**
# 

# %% [markdown]
# Estimated time needed: **45** minutes
# 

# %% [markdown]
# In this lab, we build a GenAI-powered content creation pipeline designed to transform raw research into polished, insightful blog posts.
# 
# We'll build a  CrewAI system which uses a sequential process where a Research Analyst agent gathers cutting-edge information from real-time tools like web search, and a Content Strategist agent who rewrites that information into clear, engaging content for a tech-savvy audience. We'll also create a workflow which demonstrates how autonomous agents can collaborate like human teams, moving from knowledge extraction to audience-ready content, without manual intervention.
# 
# This project is perfect for beginners who want to learn the fundamentals of multi-agent AI automation using CrewAI. You'll see how roles, tools, and tasks come together to create streamlined, intelligent workflows that save time and enhance content quality.
# 

# %% [markdown]
# ## __Table of Contents__
# 
# <ol>
#     <li><a href="#Objectives">Objectives</a></li>
#     <li>
#         <a href="#Setup">Setup</a>
#         <ol>
#             <li><a href="#Installing-Required-Libraries">Installing Required Libraries</a></li>
#         </ol>
#     </li>
#     <li><a href="#What-is-CrewAI?">What is CrewAI?</a></li>
#     <li><a href="#Setting-Up-SerperDevTool">Setting Up SerperDevTool</a></li>
#     <li><a href="#Setting-up-our-LLM">Setting up our LLM</a></li>
#     <li><a href="#Agents-in-CrewAI">Agents in CrewAI</a></li>
#     <li><a href="#Tasks-in-CrewAI">Tasks in CrewAI</a></li>
#     <li><a href="#CrewAI-Workflow">CrewAI Workflow</a></li>
#     
#     
# </ol>
# 
# <a href="#Exercises">Exercises</a>
# 

# %% [markdown]
# ## Objectives
# 
# After completing this lab, you will be able to:
# 
# - Leverage **CrewAI** to automate multi-agent workflows for intelligent content generation.  
# - Understand the **key components of CrewAI**—agents, tasks, tools, and processes—and how they work together in a sequential pipeline.  
# - Implement **real-world AI collaboration scenarios**, such as transforming technical research into reader-friendly content.    
# - Develop foundational skills to **extend and scale CrewAI workflows** across various domains like marketing, education, and research automation.
# 

# %% [markdown]
# ## Setup
# 

# %% [markdown]
# ## Required Libraries
# 
# For this lab, we will be using the following Python libraries:
# 
# * [`crewai`](https://pypi.org/project/crewai/) – The core framework for building collaborative AI workflows using agents, tasks, and process management.
# * [`crewai-tools`](https://pypi.org/project/crewai-tools/) – A set of prebuilt tools (like web search, file I/O, and APIs) that can be used by CrewAI agents.
# * [`langchain`](https://www.langchain.com/) – Provides core utilities for working with LLMs, prompts, tools, and memory management (used under the hood by CrewAI).
# * [`langchain-community`](https://pypi.org/project/langchain-community/) – Offers integration with open-source and third-party tools used in the broader LangChain ecosystem.
# 

# %% [markdown]
# ### Installing Required Libraries
# 
# The following required libraries are __not__ pre-installed in the Skills Network Labs environment. __You will need to run the following cell__ to install them:
# 

# %%
#%pip install langchain==0.3.20 | tail -n 1 
#pip install crewai==0.80.0 | tail -n 1
#%pip install langchain-community==0.3.19 | tail -n 1 
#%pip install crewai-tools==0.38.0 | tail -n 1
#%pip install databricks-sdk==0.57.0| tail -n 1

# %% [markdown]
# ----
# 

# %% [markdown]
# ## **What is CrewAI?**  
# 
# CrewAI is a **cutting-edge framework** that empowers us to create and manage teams of **autonomous AI agents** designed to collaborate on complex tasks. Think of it as our ultimate toolkit for assembling a team of virtual experts, where each member plays a **specific role**, uses **unique tools**, and works toward **clear goals**. These agents aren’t just working in isolation; they collaborate, communicate, and solve problems as a synchronized team, enabling us to achieve more than ever before.   
# 
# ### **Why CrewAI?**  
# Imagine you’re leading a project. You need specialists—each with unique expertise—who can work together to achieve a common goal. CrewAI replicates this dynamic in the world of AI by:  
# - Assigning **roles** to agents based on their purpose (e.g., a planner, an executor, or a coordinator).  
# - Equipping them with **tools** to perform their tasks efficiently.  
# - Directing them with **goals** to ensure their efforts align with the broader mission.  
# 
# This collaborative framework ensures that your AI agents can tackle challenges that are too big or too complex for a single agent to handle. Whether it's **automation**, **decision-making**, or **simulating real-world scenarios**, CrewAI empowers you to orchestrate your AI teams like never before.  
# 
# ### **How CrewAI Works**
# At its core, CrewAI provides us with a high-level framework to build “crews”—groups of role-playing agents that interact and collaborate to achieve shared objectives. Each agent is:  
# - **Assigned a Role:** Just like in a real team, every agent has a specialized function, whether it’s planning, executing, or coordinating tasks.  
# - **Equipped with Tools:** Agents are provided with the resources they need to perform their roles effectively.  
# - **Directed by Goals:** Clear objectives ensure that every agent’s efforts align with the crew’s mission.  
# 

# %% [markdown]
# ## Setting Up SerperDevTool
# 
# **What is Serper?**  
# Serper is a real-time Google Search API that allows AI agents to access up-to-date web information—effectively connecting your workflow to the latest content on the internet.
# 
# **Why are we using Serper in our workflow?**  
# Our research agent needs current, reliable information to uncover trends, breakthroughs, and insights on evolving topics like generative AI, quantum computing, or sustainability. Without web access, the agent would be limited to static, pre-trained knowledge and unable to reflect the latest developments.
# 
# To use the `SerperDevTool`, it requires an **API key**. This key grants access to the web search service and allows our agents to fetch real-time data during execution.
# 
# > You will need to obtain your API Key from [serper.dev](https://serper.dev).  
# > - Sign up or log in with your email  
# > - Navigate to the **Dashboard**  
# > - Click on **API Keys**  
# > - Copy the key and replace `API_KEY` in your code with the value provided
# 
# To learn more about the `SerperDevTool` and its capabilities, visit the [official documentation](https://serper.dev/).
# 

# %% [markdown]
# Enter  API key 
# 

# %%
import os 
os.environ['SERPER_API_KEY'] = "API_KEY_HERE"

# %% [markdown]
# Import ```SerperDevTool``` from ```crewai_tools```. 
# 

# %%
#%%capture

from crewai_tools import SerperDevTool

# %% [markdown]
# Initialize the SerperDev search tool  object (requires an API key)
# 

# %%
search_tool=SerperDevTool()
print(type(search_tool))

# %% [markdown]
# Run a search query 
# 

# %%
search_query = "Latest Breakthroughs in machine learning"
search_results =search_tool.run(query=search_query )

# Print the results
print(f"Search Results for '{search_results}':\n")

# %% [markdown]
# The ````search_results```` dictionary has a lot of info, so here is an overview of what each key contains:
# 
# 
# - **searchParameters**: Query metadata (term, engine, result count)
# - **organic**: Search results (title, link, snippet, position)
# - **peopleAlsoAsk**: Related questions with answers
# - **relatedSearches**: Alternative search queries
# - **credits**: API usage tracking
# 

# %%
print("keys of search_results", search_results.keys())

# %% [markdown]
# ## Setting up our LLM
# 
# Next, we need to set up our **LLM (Large Language Model)**—this can be **any model** based on our needs. Here, we are going to use **Meta Llama 3.3 70b instruct**. The choice of model depends on factors such as **accuracy, speed, and recipe understanding** for our meal planning tasks.
# 

# %%
from crewai import LLM

llm = LLM(
        model="watsonx/meta-llama/llama-3-3-70b-instruct",
        base_url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        max_tokens=2000,
)

# %% [markdown]
# ## **Agents in CrewAI**  
# 
# In CrewAI, **agents** are the foundational units of any multi-agent system. Each agent is designed to perform a specific role, solve tasks autonomously, and collaborate seamlessly with other agents. They’re more than mere programs—they are your specialized team members in an AI-powered ecosystem.  
# 
# ---
# 

# %% [markdown]
# 
# 
# A CrewAI agent isn’t just a block of code; it’s a thoughtfully designed entity with the following parameters:  
# 
# 1. **Role**  
#    An agent’s role defines its purpose in the system. Roles are as diverse as your project needs, such as a **"Data Researcher"** hunting for insights or a **"Reporting Analyst"** preparing comprehensive summaries.  
# 
# 2. **Goal**  
#    Each agent operates with a defined goal—a guiding star that shapes its decisions and actions. For instance, an agent with the goal to **“Uncover cutting-edge developments in AI”** will consistently align its behavior to fulfill this objective.  
# 
# 3. **Backstory**  
#    An agent’s backstory is like its resume, providing context or personality that influences how it behaves and interacts. For example, a seasoned **“Senior Data Researcher”** with years of experience might approach tasks differently from a **“Junior Analyst”** just starting out. This feature adds depth and relatability to agent interactions, making them more dynamic and tailored.  
# 
# 
# 4. **Tools**  
#    Just like any professional needs the right tools to excel, agents in CrewAI are equipped with specialized tools to boost their performance. Whether it’s a **web search utility** for gathering information, a **data analysis engine** for crunching numbers, or an **API connector** to integrate external services, tools expand an agent’s capabilities. The right tool can help an agent complete its tasks more efficiently and effectively, enabling it to work smarter, not harder.  
# 
# 5. **Configuration**  
#    Agents in CrewAI are configured using simple YAML files, offering a modular, readable, and scalable approach to defining their attributes. This makes setting up agents intuitive, even for large systems ( in this tutroal we will not use a YML files 
# 
# 
# 

# %% [markdown]
# ####  **Defining an Agent Directly as a Python Object**
# For more flexibility or when working in a programmatic environment, you can define agents directly in your code. This approach allows you to quickly integrate dynamic parameters and logic into the agent’s setup.
# 
# In this section, we're defining the research agent which will gather and analyze information from the web. This "Senior Research Analyst" uses the SerperDevTool to search for relevant content, working independently without delegation. The agent serves as the first step in our workflow, collecting the raw data that other agents will later refine and present.
# 
# Example of defining an agent in **Python**:
# 
# 

# %%
from crewai import Agent

research_agent = Agent(
  role='Senior Research Analyst',
  goal='Uncover cutting-edge information and insights on any subject with comprehensive analysis',
  backstory="""You are an expert researcher with extensive experience in gathering, analyzing, and synthesizing information across multiple domains. 
  Your analytical skills allow you to quickly identify key trends, separate fact from opinion, and produce insightful reports on any topic. 
  You excel at finding reliable sources and extracting valuable information efficiently.""",
  verbose=True,
  allow_delegation=False,
  llm = llm,
  tools=[SerperDevTool()]
)

# %% [markdown]
# 
# In this Python example, an agent is created with the same role, goal, backstory, and tools as the YAML example. However, this method allows you to easily pass in dynamic variables and parameters at runtime, making it ideal for scenarios where the agent configuration needs to change dynamically.
# 

# %%
research_agent

# %% [markdown]
# In CrewAI, we use multiple specialized agents to complete complex tasks through collaboration. In our research-report example:
# 
# 1. We created a **Researcher Agent** that gathers information
# 2. Now we will create a **Writer Agent** that takes the output from our Researcher Agent
# 3. The Writer transforms research findings into well-structured content for the target audience
# 
# Let's create the writer agent with the following parameters:
# 
# * **role**: 'Tech Content Strategist' - Job function within the workflow
# * **goal**: 'Craft well-structured and engaging content based on research findings' - The agent's specific objective
# * **backstory**: Background that shapes the agent's approach and style
# * **verbose**: True - Controls logging detail level
# * **allow_delegation**: True - Enables task assignment to other agents
# 

# %%
# Define your agents with roles and goals
# Define the Writer Agent
writer_agent = Agent(
  role='Tech Content Strategist',
  goal='Craft well-structured and engaging content based on research findings',
  backstory="""You are a skilled content strategist known for translating 
  complex topics into clear and compelling narratives. Your writing makes 
  information accessible and engaging for a wide audience.""",
  verbose=True,
  llm = llm,
  allow_delegation=True
)

# %%
writer_agent 

# %% [markdown]
# ## **Tasks in CrewAI**
# Tasks are like to-do items for our AI agents. Each task has specific instructions, details, and tools for the agent to follow and complete the job.
# 
# For example:
# - A task could ask an agent to "research the latest AI trends."
# - Another task could ask a different agent to "write a detailed report based on the research."
# 
# 

# %% [markdown]
# Here is an outline of the porcess:
# 
# 1. **Define agents** with their roles, goals, and tools
# 2. **Create tasks** and assign them to specific agents
# 3. **Combine agents and tasks** into a Crew with an execution process
# 

# %% [markdown]
# #### **How Tasks Work**  
# There are two ways tasks can run:  
# 
# 1. **Sequential**: Tasks are executed one after the other, like following a recipe step-by-step. Each task waits for the previous one to finish.  
# 2. **Hierarchical**: Tasks are assigned based on agent skills or roles, and multiple tasks can run in parallel if they don’t depend on each other.  
# 
# 
# 

# %% [markdown]
# #### **What Can a Task Include?**
# Each task has these details:
# - **Description**: What needs to be done.
# - **Expected Output**: What the result should look like.
# - **Agent**: Who’s responsible for the task.
# - **Tools**: The tools the agent can use for this task.
# - **Context**: Outputs from other tasks that help this task.
# - **Async Execution**: Whether the task runs in the background or not.
# - **Output Format**: Whether the results are plain text, JSON, or a structured model.
# 

# %% [markdown]
# Here's how we set up a Crew (our team of agents) and tasks in code `research_task` and `writer_task`. 
# 
# In this step, we define a Task for the Researcher Agent. This task will involve gathering and analyzing key insights on any topic specified through the `{topic}` parameter. The agent will use the SerperDevTool to uncover major trends, identify new technologies, and evaluate their effects on the industry. This flexible approach allows us to research different subjects by simply changing the input parameter when kicking off the crew.
# 

# %%
from crewai import Task

research_task = Task(
  description="Analyze the major {topic}, identifying key trends and technologies. Provide a detailed report on their potential impact.",
  agent=research_agent,
  expected_output="A detailed report on {topic}, including trends, emerging technologies, and their impact."
)

# %% [markdown]
# Now, we will define the task for the Writer Agent, who will take the research findings and transform them into a well-structured article. The Writer Agent will ensure the content is engaging, informative, and easy to understand, making complex topics more accessible.
# 

# %%
# Create a task for the Writer Agent
writer_task = Task(
  description="Create an engaging blog post based on the research findings about {topic}. Tailor the content for a tech-savvy audience, ensuring clarity and interest.",
  agent=writer_agent,
  expected_output="A 4-paragraph blog post on {topic}, written clearly and engagingly for tech enthusiasts."
)

# %% [markdown]
# ## CrewAI Workflow
# 
# The  `Crew` object, which is the central orchestration mechanism in CrewAI. This crew brings together our specialized agents and their assigned tasks into a cohesive workflow.
# 
# The `Crew` constructor takes several important parameters:
# - `agents`: A list of the AI agents that will be part of this crew ```research_agent``` abd  ```writer_agent```
# - `tasks`: A list of specific tasks these agents will perform ```research_task``` and ```writer_task```
# - `process`: Defines how tasks will be executed - in this case `Process.sequential means tasks will run one after another in the specified order (research first, then writing)
# - `verbose`: When set to `True`, this enables detailed logging, making it easier to follow the crew's execution and troubleshoot any issues
# 
# Once configured, you can start the entire workflow with a single command: `crew.kickoff()`, which will execute the tasks in sequence and return the final results.
# 

# %%
from crewai import Crew, Process

crew = Crew(
    agents=[research_agent, writer_agent],
    tasks=[research_task, writer_task],
    process=Process.sequential,
    verbose=True 
)

# %% [markdown]
# The method ```kickoff()``` sets everything rolling - it starts all your agents working on their tasks and returns the results when they're done. By using ```inputs={"topic": "quantum computing breakthroughs of 2024"}```, we can specify exactly what subject our agents should research, making our system flexible enough to analyze any topic without changing the task definitions.
# 

# %%
result = crew.kickoff(inputs={"topic": "Latest Generative AI breakthroughs"})

# %% [markdown]
# The result is a ```crew_output``` 
# 

# %%
type(result)

# %%
result

# %% [markdown]
# The `result.raw` output text contains the final content produced by our last agent in the workflow. We can easily access this text to see the complete results:
# 

# %%
final_output = result.raw
print("Final output:", final_output)

# %% [markdown]
# The `tasks_output` list gives us access to outputs from each task in the order they were executed:
# 

# %%
tasks_outputs = result.tasks_output

# %% [markdown]
# We see the output of the research task object. This lets us access both the task description and the content the agent produced:
# 

# %%
print("Task Description", tasks_outputs[0].description)
print("Output of research task ",tasks_outputs[0])

# %% [markdown]
# We also have the description and output for the writer task using the raw property:
# 

# %%
print("Writer task description:", tasks_outputs[1].description)
print(" \nOutput of writer task:", tasks_outputs[1].raw)

# %% [markdown]
# 
# In addition to the task output, we can access the agent that performed each task:
# 

# %%
print("We can get the agent for researcher task:  ",tasks_outputs[0].agent)
print("We can get the agent for the writer task: ",tasks_outputs[1].agent)

# %% [markdown]
# ---
# After your agents complete their tasks, CrewAI provides detailed performance metrics that help you monitor resource usage and optimize your multi-agent systems. Token usage analytics are particularly important as they directly impact operational costs and system efficiency.
# 

# %%
token_count = result.token_usage.total_tokens
prompt_tokens = result.token_usage.prompt_tokens
completion_tokens = result.token_usage.completion_tokens

print(f"Total tokens used: {token_count}")
print(f"Prompt tokens: {prompt_tokens} (used for instructions to the model)")
print(f"Completion tokens: {completion_tokens} (generated in response)")

# %% [markdown]
# ## Exercises 
# In these exercises, you will create a web publishing component for your fact-checking application by implementing a web designer agent and task. This final piece will transform the analyzed and written content into a professional webpage that presents verification results clearly to users.
# 

# %% [markdown]
# ### Exercise 1: Create a Social Media Strategist Agent
# 
# Create a Social Media Agent which curates a summary and a short-form version (such as tweets or LinkedIn posts).
# 

# %%
social_media_agent = Agent(
  role='Social Media Strategist Agent',
  goal='Curates a summary and short-form version of social media posts',
  backstory="""You are an expert social media analyst with extensive experience in gathering, analyzing, and synthesizing information across different types of social media posts. 
  Your analytical skills allow you to quickly identify key trends, separate fact from opinion, and produce summaries on any social media post. 
  You excel at extracting valuable information efficiently, making complex information easy to understand and attractive to readers.""",
  verbose=True,
  allow_delegation=False,
  llm = llm,
  tools=[SerperDevTool()]
)

# %% [markdown]
# <details>
#     <summary>Click here for the solution</summary>
# 
# ```python
# 
# social_agent = Agent(
#     role='Social Media Strategist',
#     goal='Generate engaging social media snippets based on the full article',
#     backstory="A digital storyteller who excels at crafting compelling posts to drive engagement and traffic.",
#     verbose=True
# )
# 
# 
# ```
# 
# </details>
# 

# %% [markdown]
# ### Exercise 2: Defining a Social Media Strategy Task
# 
# Create a task for the Social Media Strategist agent to generate engaging and platform-specific posts (such as LinkedIn or X/Twitter) based on the research and blog content. This agent will help amplify the reach of your content by distilling key insights into short, compelling messages.
# 

# %%
social_media_strategy_task = Task(
  description="Generate engaging and platform-specific posts (such as LinkedIn or X/Twitter) based on the research and blog content.",
  agent=social_media_agent,
  expected_output="A clear, concise, attention-grabbing social media caption on {topic}, highlight the key insights from blog content."
)

# %% [markdown]
# <details>
#     <summary>Click here for the solution</summary>
# 
# ```python
# social_task = Task(
#     description=(
#         "Summarize the blog post about {topic} into 2–3 engaging social media posts "
#         "suitable for platforms like LinkedIn or Twitter. Make sure the tone is informative, "
#         "professional, and encourages further reading."
#     ),
#     agent=social_agent,
#     expected_output="A series of 2–3 well-written social posts highlighting the key insights from the blog content."
# )
# ```
# 
# </details>
# 

# %% [markdown]
# ### Exercise 3: Create a Complete Crew Object 
# 
# Include research, writing, and social media agents along with their tasks, configured for sequential processing with verbose output and apply the method ```kickoff()``` method.
# 

# %%
crew = Crew(
    agents=[research_agent, writer_agent, social_media_agent],
    tasks=[research_task, writer_task, social_media_strategy_task],
    process=Process.sequential,
    verbose=True 
)

result = crew.kickoff(inputs={"topic": "Latest Generative AI Breakthroughs"})

# %% [markdown]
# <details>
#     <summary>Click here for the solution</summary>
# 
# ```python
# crew = Crew(
#     agents=[research_agent, writer_agent, social_agent],
#     tasks=[research_task, writer_task, social_task],
#     process=Process.sequential,  # Tasks will be executed one after another
#     verbose=True
# )
# 
# # Run the crew and capture the final output (includes research, blog post, and social media content)
# result = crew.kickoff(inputs={"topic": "Latest Generative AI breakthroughs"})
# ```
# 
# </details>
# 

# %% [markdown]
# ## Authors
# 

# %% [markdown]
# [Karan Goswami](https://author.skills.network/instructors/karan_goswami)
# 
# [Kunal Makwana](https://author.skills.network/instructors/kunal_makwana)
# 

# %% [markdown]
# ## Change Log
# 
# <details>
#     <summary>Click here for the changelog</summary>
# 
# |Date (YYYY-MM-DD)|Version|Changed By|Change Description|
# |-|-|-|-|
# |2025-07-17|0.1|Karan Goswami|Initial version created|
# |2025-07-22|0.2|Steve Ryan|ID review|
# 
# </details>
# 
# ---
# 
# 
# Copyright © IBM Corporation. All rights reserved.
# 


