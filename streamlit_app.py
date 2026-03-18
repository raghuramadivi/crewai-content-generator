import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Content Research & Writer",
    page_icon="📝",
    layout="wide"
)

st.title("📝 Content Research & Writer")
st.markdown("Powered by **CrewAI** · Multi-Agent Pipeline · GPT-4 + Serper")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Content Settings")

    topic = st.text_area(
        "Enter the topic",
        height=100,
        placeholder="Enter the topic you want to research and write about..."
    )

    st.markdown("#### LLM Settings")
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Higher = more creative. Lower = more focused and factual."
    )

    with st.expander("How to use"):
        st.markdown("""
        1. Enter your desired content topic in the text area above.
        2. Play with the **temperature** slider to control creativity.
        3. Click **Generate Content** to start the multi-agent pipeline.
        4. Download the result as a Markdown file when done.
        """)

# ── Core generation function ──────────────────────────────────────────────────
def generate_content(topic: str, temperature: float = 0.1):
    """Run the two-agent CrewAI pipeline and return the result."""

    # Tools
    llm = LLM(model="gpt-4", temperature=temperature)
    search_tool = SerperDevTool(n_results=10)

    # Agent 1 — Senior Research Analyst
    researcher = Agent(
        role="Senior Research Analyst",
        goal=(
            f"Research, analyze and synthesize comprehensive information on {topic} "
            "including recent developments, key trends, and expert insights."
        ),
        backstory=(
            "You are an expert research analyst with advanced web research skills. "
            "You excel at finding, analyzing, and synthesizing information from across "
            "the internet using the search tool. You are skilled at distinguishing "
            "reliable resources from unreliable ones, fact-checking, cross-referencing "
            "information, and identifying key patterns and insights. "
            "You provide well-organized research with all citations."
        ),
        tools=[search_tool],
        llm=llm,
        allow_delegation=False,
        verbose=True
    )

    # Agent 2 — Content Writer
    content_writer = Agent(
        role="Content Writer",
        goal=(
            f"Transform research findings into an engaging, accurate blog post "
            f"about {topic} while maintaining all factual accuracy and citations."
        ),
        backstory=(
            "You are a skilled content writer specialized in creating engaging, "
            "accessible content from technical research. You work closely with the "
            "Senior Research Analyst and excel at maintaining the perfect balance "
            "between informative and entertaining writing, while ensuring all facts "
            "and citations from the research are preserved. You have a talent for "
            "making complex topics approachable without oversimplifying."
        ),
        tools=[],
        llm=llm,
        allow_delegation=False,
        verbose=True
    )

    # Task 1 — Research
    research_task = Task(
        description=(
            f"Conduct comprehensive research on: {topic}\n\n"
            "Your research must include:\n"
            "- Recent developments and news\n"
            "- Key industry trends and innovations\n"
            "- Expert opinions and analysis\n"
            "- Statistical data and market insights\n"
            "- Evaluate sources for credibility and fact-check all information\n"
            "- Organize findings into a structured research brief\n"
            "- Include all relevant citations and source URLs"
        ),
        expected_output=(
            "A detailed research report containing:\n"
            "- Executive summary of key findings\n"
            "- Comprehensive analysis of current trends and developments\n"
            "- List of verified facts and statistics\n"
            "- All citations and links to original sources\n"
            "- Clear categorization of main themes and patterns\n"
            "Formatted with clear sections and bullet points for easy reference."
        ),
        agent=researcher
    )

    # Task 2 — Writing
    writing_task = Task(
        description=(
            f"Transform the technical research into an accessible, engaging blog post about {topic}.\n\n"
            "Requirements:\n"
            "- Maintain all factual accuracy and citations from the research\n"
            "- Write an attention-grabbing introduction\n"
            "- Use well-structured body sections with clear headings\n"
            "- Preserve all citation source URLs in Markdown link format\n"
            "- Make complex information approachable without oversimplifying\n"
            "- Follow Markdown formatting strictly"
        ),
        expected_output=(
            "A polished blog post in Markdown format that:\n"
            "- Engages readers while maintaining accuracy\n"
            "- Is properly structured with clear sections\n"
            "- Includes inline citations hyperlinked to original source URLs\n"
            "- Presents information in an accessible yet informative way\n"
            "- Uses H1 for the title and H3 for subsections\n"
            "- Is ready to publish on any blogging platform"
        ),
        agent=content_writer
    )

    # Crew
    crew = Crew(
        agents=[researcher, content_writer],
        tasks=[research_task, writing_task],
        verbose=True
    )

    result = crew.kickoff(inputs={"topic": topic})
    return result

# ── Main UI ───────────────────────────────────────────────────────────────────
if st.button("🚀 Generate Content", type="primary", use_container_width=True):
    if not topic.strip():
        st.warning("Please enter a topic before generating.")
    else:
        with st.spinner("Running multi-agent pipeline... This may take a few minutes."):
            try:
                result = generate_content(topic, temperature)

                st.success("✅ Content generated successfully!")
                st.markdown("---")
                st.markdown(str(result))

                # Download button
                st.download_button(
                    label="⬇️ Download as Markdown",
                    data=str(result),
                    file_name=f"{topic[:40].replace(' ', '_').lower()}_article.md",
                    mime="text/markdown",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.info("Check your API keys in the .env file and try again.")