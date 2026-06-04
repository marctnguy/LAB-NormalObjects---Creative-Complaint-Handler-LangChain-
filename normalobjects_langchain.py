import os
import random
from typing import Dict, List

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_openai import ChatOpenAI


load_dotenv()


# Step 1: initialize the model from the .env-loaded environment.
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY is missing. Add it to your .env file before running this script."
    )

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

tracker = None


def record_tool_usage(tool_name: str) -> None:
    """Record a tool call when the tracker has been initialized."""
    if tracker is not None:
        tracker.track_usage(tool_name)


# Step 2: creative tools for the Normal Objects universe.
@tool
def consult_demogorgon(complaint: str) -> str:
    """Get the Demogorgon's perspective on a complaint about the Upside Down.
    The Demogorgon is a creature from the Upside Down. It might have insights
    about interdimensional inconsistencies, but its perspective is... unique.
    
    Args:
        complaint: The complaint about the Upside Down
        
    Returns:
        The Demogorgon's perspective (creative and possibly chaotic)
    """
    responses = [
        f"The Demogorgon tilts its head. It seems confused by '{complaint}'. Perhaps the issue is that you're thinking in three dimensions?",
        f"The Demogorgon makes a sound that might be agreement. It suggests that the problem might be temporal - things work differently in the Upside Down's time.",
        f"The Demogorgon appears to be eating something. It doesn't seem to understand the concept of '{complaint}' - maybe consistency isn't a priority there?",
    ]
    record_tool_usage("consult_demogorgon")
    import random
    return random.choice(responses)


@tool
def check_hawkins_records(query: str) -> str:
    """Search Hawkins historical records for information.
    Hawkins has a long history of strange occurrences. These records
    might contain clues about patterns or explanations.
    
    Args:
        query: What to search for in the records
        
    Returns:
        Information from Hawkins historical records
    """
    records = {
        "portal": "Records show portals have opened on various dates with no clear pattern. Weather, electromagnetic activity, and unknown factors seem involved.",
        "monsters": "Historical records indicate creatures from the Upside Down behave differently based on environmental factors, time of day, and proximity to certain individuals.",
        "psychics": "Records show that psychic abilities vary greatly. Some individuals can move objects but not see the future, others can see visions but not move things.",
        "electricity": "Hawkins has a history of electrical anomalies. Records suggest a connection between the Upside Down and electromagnetic fields.",
    }

    for key, value in records.items():
        if key in query.lower():
            record_tool_usage("check_hawkins_records")
            return value

    record_tool_usage("check_hawkins_records")
    return (
        f"Records don't contain specific information about '{query}', but they note that many unexplained events have occurred in Hawkins over the years."
    )


@tool
def cast_interdimensional_spell(problem: str, creativity_level: str = "medium") -> str:
    """Suggest a creative interdimensional spell to fix a problem.
    Sometimes the best solution is a creative one that doesn't follow normal rules.
    This tool suggests imaginative fixes for Upside Down problems.
    
    Args:
        problem: The problem to solve
        creativity_level: How creative to be (low, medium, high)
        
    Returns:
        A creative spell or solution suggestion
    """
    creativity_multiplier = {"low": 1, "medium": 2, "high": 3}[creativity_level]

    spells = [
        f"Try chanting 'Becma Becma Becma' three times while holding a Walkman. This might recalibrate the interdimensional frequencies related to: {problem}",
        f"Create a salt circle and place a compass in the center. The magnetic anomalies might help stabilize: {problem}",
        f"Play 'Running Up That Hill' backwards at the exact location of the issue. The temporal resonance could fix: {problem}",
        f"Gather three items: a lighter, a compass, and something personal. Arrange them in a triangle while thinking about: {problem}. The emotional connection might help.",
    ]
    record_tool_usage("cast_interdimensional_spell")
    import random
    selected = random.sample(spells, min(creativity_multiplier, len(spells)))
    return "\n".join(selected)


@tool
def gather_party_wisdom(question: str) -> str:
    """Ask the D&D party (Mike, Dustin, Lucas, Will) for their collective wisdom.
    The party has solved many mysteries together. Their combined knowledge
    and different perspectives can provide insights.
    
    Args:
        question: The question or problem to ask the party about
        
    Returns:
        The party's collective wisdom and suggestions
    """
    party_responses = {
        "portal": "Mike: 'Portals are unpredictable, but they usually open near strong emotional events or electromagnetic disturbances.' Dustin: 'Also, they seem to follow some kind of pattern related to the Mind Flayer's activity.'",
        "monsters": "Lucas: 'Demogorgons are territorial but also opportunistic.' Will: 'They can sense fear and strong emotions. Maybe that's why they act differently sometimes.'",
        "psychics": "Mike: 'El's powers seem connected to her emotional state.' Dustin: 'And they're limited by her physical and mental energy. That's probably why she can't do everything.'",
        "electricity": "Lucas: 'The Upside Down seems to interfere with electrical systems.' Dustin: 'But it also creates strange connections. It's like a feedback loop.'",
    }

    for key, response in party_responses.items():
        if key in question.lower():
            record_tool_usage("gather_party_wisdom")
            return response

    record_tool_usage("gather_party_wisdom")
    return (
        "The party huddles together. Mike: 'This is a tough one.' Dustin: 'We need more information.' Lucas: 'Let's think about what we know.' Will: 'Maybe we should consult other sources?'"
    )


tools = [
    consult_demogorgon,
    check_hawkins_records,
    cast_interdimensional_spell,
    gather_party_wisdom,
]


class ToolUsageTracker:
    """Track tool usage for analysis."""

    def __init__(self):
        self.usage_count = {tool.name: 0 for tool in tools}
        self.tool_sequences = []

    def track_usage(self, tool_name: str):
        """Track when a tool is used."""
        if tool_name in self.usage_count:
            self.usage_count[tool_name] += 1
            self.tool_sequences.append(tool_name)

    def get_statistics(self):
        """Get usage statistics."""
        return {
            "total_tool_calls": sum(self.usage_count.values()),
            "tool_counts": self.usage_count,
            "most_used": max(self.usage_count.items(), key=lambda x: x[1])[0]
            if self.usage_count
            else None,
            "tool_sequences": self.tool_sequences,
        }

    def merge_from(self, other: "ToolUsageTracker") -> None:
        """Merge counts and sequences from another tracker into this one."""
        for tool_name, count in other.usage_count.items():
            self.usage_count[tool_name] += count
        self.tool_sequences.extend(other.tool_sequences)


tracker = ToolUsageTracker()
aggregate_tracker = ToolUsageTracker()

tool_names: List[str] = [tool_obj.name for tool_obj in tools]
tool_descriptions: Dict[str, str] = {
    tool_obj.name: tool_obj.description for tool_obj in tools
}

print(f"Created {len(tools)} creative tools:")
for tool_obj in tools:
    print(f"  - {tool_obj.name}: {tool_obj.description[:60]}...")


# Step 3: build the prompt, agent, and executor.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are Becma's Chaos Mode, a creative complaint handler for the Normal Objects universe. "
            "Use the available tools freely, combine them in any order, and answer in an imaginative but clear way. "
            "If the complaint is unclear, consult more than one source before answering.",
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# Step 4: sample complaints and a helper for testing.
complaints: List[str] = [
    "Why do demogorgons sometimes eat people and sometimes don't?",
    "The portal opens on different days—is there a schedule?",
    "Why can some psychics see the Downside Up and others can't?",
    "Why do creatures and power lines react so strangely together?",
]


def handle_complaint(complaint: str) -> str:
    """Handle a single complaint."""
    print(f"\n{'=' * 60}")
    print(f"COMPLAINT: {complaint}")
    print(f"{'=' * 60}\n")

    result = agent_executor.invoke({"input": complaint})
    return result["output"]


# Step 5: Analyze Agent Behavior
if __name__ == "__main__":
    print("Testing agent with sample complaints...\n")

    analysis_runs = 3
    for run_number in range(1, analysis_runs + 1):
        print(f"\n=== Analysis Run {run_number} ===")
        current_tracker = ToolUsageTracker()
        tracker = current_tracker

        try:
            for complaint in complaints[:2]:
                response = handle_complaint(complaint)
                print(f"\nRESPONSE: {response}\n")
        finally:
            run_stats = current_tracker.get_statistics()
            aggregate_tracker.merge_from(current_tracker)
            print("\n--- Run Summary ---")
            print(f"Total tool calls: {run_stats['total_tool_calls']}")
            print(f"Tool usage counts: {run_stats['tool_counts']}")
            print(f"Most used tool: {run_stats['most_used']}")
            print("Tool sequence examples:")
            for i in range(min(3, len(run_stats["tool_sequences"]))):
                print(
                    f"  Sequence {i + 1}: {' -> '.join(run_stats['tool_sequences'][i:i+3])}"
                )

    print("\n=== Aggregated Tool Usage Analysis ===")
    stats = aggregate_tracker.get_statistics()
    print(f"Total tool calls: {stats['total_tool_calls']}")
    print(f"Tool usage counts: {stats['tool_counts']}")
    print(f"Most used tool: {stats['most_used']}")
    print("\nTool sequence examples:")
    for i in range(min(3, len(stats["tool_sequences"]))):
        print(f"  Sequence {i + 1}: {' -> '.join(stats['tool_sequences'][i:i+3])}")
