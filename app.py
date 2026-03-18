from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

load_dotenv()

topic="Medical Industry using Generative AI"

#Tool 1
llm=LLM(model="gpt-4")

#Tool 2
search_tool=SerperDevTool(n=10)

#Agent 1
senior_research_analyst= Agent(
    role="Senior Research Analyst",
    goal="Research, analyse, and synthesize comprehensive information on {topic} from reliable web sources",
    backstory="""You are an expert research analyst with advanced web research skills. 
                You excel at finding, analyzing, and synthesizing information from 
                across the internet using the Search tool. You are skilled at 
                distinguishing reliable resources from unreliable ones, fact-checking, 
                cross-referencing information, and identifying key patterns and insights. 
                You provide well-organized research with all relevant citations and 
                source links.""",
    allow_delegation=False,
    verbose=True,
    tools=[search_tool],
    llm=llm

)

#Agent 2 Content Wrtier

content_writer=Agent(
    role="Content Writer",
    goal= "transform research findings into engaging blog posts while maintaining accuracy",
    backstory="""You are a skilled content writer specialized in creating engaging, 
            accessible content from technical research. You work closely with the 
            senior research analyst and excel at maintaining the perfect balance 
            between informative and entertaining writing, while ensuring all facts 
            and citations from the research are preserved. You have a talent for 
            making complex topics approachable without oversimplifying.""",
    allow_delegation=False,
    verbose=True,
    llm=llm        
)

# Research Tasks
research_tasks=Task(
    description=(
        f"""
        Conduct comprehensive research on {topic} including:
        
        1. Recent developments and news
        2. Key industry trends and innovations
        3. Expert opinions and analysis
        4. Statistical data and market insights
        
        Evaluate for credibility and fact check 
        all information.
        
        Organize findings into a structured 
        research brief.
        
        Include all relevant citations and 
        resources.
    """),
    expected_output="""
        A detailed research report containing:
        
        - Executive summary of key findings
        - Comprehensive analysis of current 
          trends and developments
        - List of verified facts and statistics
        - All citations and links to original 
          sources
        - Clear categorization of main themes 
          and patterns
        
        Please format with clear sections and 
        bullet points for easy reference.
    """,
    agent=senior_research_analyst
)

#Task 2 Content Writing
writing_task=Task(
    description="""
    Transform the research findings into an engaging blog post:
    
    1. Transform technical information into accessible content
    2. Maintain all factual accuracy and citations from the research
    3. Write an attention-grabbing introduction
    4. Create well-structured body sections with clear headings
    5. Preserve all citations in source URL format
    6. Use simple language while keeping the content informative
    7. Follow proper markdown formatting throughout
    """,
    expected_output="""
    A polished blog post in markdown format that:
    
    - Engages readers while maintaining accuracy
    - Is properly structured with clear sections
    - Includes inline citations hyperlinked to original source URLs
    - Presents information in an accessible yet informative way
    - Follows markdown formatting:
        - Use H1 for the main title
        - Use H3 for subsections
    """,
    agent=content_writer
)

crew=Crew(
    agents= [senior_research_analyst,content_writer],
    tasks= [research_tasks,writing_task],
    verbose=True
)

result= crew.kickoff(inputs={"topic": topic})