# test_research_crew.py
from guide_creator_flow.crews.research_crew import ResearchCrew

def test_research():
    result = ResearchCrew().kickoff(inputs={
        "topic": "What is a Research Agent?",
        "audience_level": "beginner"
    })
    
    print("\n" + "="*50)
    print("RESEARCH RESULTS")
    print("="*50)
    print(result.raw)
    print("\n" + "="*50)

if __name__ == "__main__":
    test_research()