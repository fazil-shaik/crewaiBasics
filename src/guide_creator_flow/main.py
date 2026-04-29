#!/usr/bin/env python
import json
import os
import re
from typing import List, Dict
from pydantic import BaseModel, Field
from crewai import LLM
from crewai.flow.flow import Flow, listen, start
from guide_creator_flow.crews.content_crew.content_crew import ContentCrew
from guide_creator_flow.crews.research_crew.research_crew import ResearchCrew

# Define our models for structured data
class Section(BaseModel):
    title: str = Field(description="Title of the section")
    description: str = Field(description="Brief description of what the section should cover")



class GuideOutline(BaseModel):
    title: str = Field(description="Title of the guide")
    introduction: str = Field(description="Introduction to the topic")
    target_audience: str = Field(description="Description of the target audience")
    sections: List[Section] = Field(description="List of sections in the guide")
    conclusion: str = Field(description="Conclusion or summary of the guide")

# Define our flow state
class GuideCreatorState(BaseModel):
    topic: str = ""
    audience_level: str = ""
    guide_outline: GuideOutline = None
    sections_content: Dict[str, str] = {}

class GuideResearchState(BaseModel):
    topic: str = ""
    audience_level: str = ""
    research_findings: Dict[str, str] = {}


class GuideResearchFlow(Flow[GuideResearchState]):
    """Flow for researching a topic to gather information and insights for the guide"""

    @start()
    def get_research_input(self):
        """Get input from the user about the research topic and audience"""
        print("\n=== Research Your Guide Topic ===\n")

        # Get user input
        self.state.topic = input("What topic would you like to research for your guide? ")

        # Get audience level with validation
        while True:
            audience = input("Who is your target audience? (beginner/intermediate/advanced) ").lower()
            if audience in ["beginner", "intermediate", "advanced"]:
                self.state.audience_level = audience
                break
            print("Please enter 'beginner', 'intermediate', or 'advanced'")

        print(f"\nResearching {self.state.topic} for {self.state.audience_level} audience...\n")
        return self.state
    
    @listen(get_research_input)
    def perform_research(self, state):
        """Perform research using the research crew to gather information and insights"""
        print("🔍 Gathering comprehensive research...")
        print("-" * 50)

        try:
            # Run the research crew
            result = ResearchCrew().kickoff(inputs={
                "topic": state.topic,
                "audience_level": state.audience_level
            })
            
            # Store research findings in state
            # The result contains both raw research and synthesized findings
            self.state.research_findings = {
                "raw": result.raw if hasattr(result, 'raw') else str(result),
                "synthesized": result.synthesized_findings if hasattr(result, 'synthesized_findings') else ""
            }
            
            print(f"\n  Research completed for {state.topic}")
            print(f"   Research saved to state for guide creation")
            
            # Optional: Save research to file for reference
            os.makedirs("output/research", exist_ok=True)
            with open(f"output/research/research_{state.topic.replace(' ', '_')}.md", "w") as f:
                f.write(f"# Research: {state.topic}\n\n")
                f.write(f"**Audience Level:** {state.audience_level}\n\n")
                f.write("## Raw Research\n\n")
                f.write(self.state.research_findings.get("raw", ""))
                f.write("\n\n## Synthesized Findings\n\n")
                f.write(self.state.research_findings.get("synthesized", ""))
            
            print(f"   Research saved to output/research/")
            return self.state.research_findings
            
        except Exception as e:
            print(f" Error performing research: {e}")
            import traceback
            traceback.print_exc()
            self.state.research_findings = {
                "error": str(e),
                "raw": f"Research failed: {e}",
                "synthesized": ""
            }
            return self.state.research_findings

        

class GuideCreatorFlow(Flow[GuideCreatorState]):
    """Flow for creating a comprehensive guide on any topic"""

    def repair_json(self, json_str: str) -> str:
        """Attempt to repair common JSON issues"""
        # Remove control characters
        json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
        
        # Fix missing commas between objects in arrays
        json_str = re.sub(r'}\s*{', '},{', json_str)
        
        # Fix missing quotes around keys
        json_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
        
        # Fix trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*\]', ']', json_str)
        
        # Fix single quotes
        json_str = json_str.replace("'", '"')
        
        return json_str

    def extract_json_from_response(self, response: str) -> dict:
        """Extract JSON from LLM response with multiple fallback strategies"""
        
        # Strategy 1: Try to find JSON between braces
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            
            # Try parsing as-is
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try repairing
                try:
                    repaired = self.repair_json(json_str)
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass
        
        # Strategy 2: Try to extract using more aggressive pattern
        # Look for content that might be missing outer braces
        if '"title"' in response and '"sections"' in response:
            # Try to wrap in braces if missing
            if not response.strip().startswith('{'):
                response = '{' + response + '}'
                try:
                    return json.loads(response)
                except:
                    pass
        
        # Strategy 3: Use the LLM to fix its own JSON
        print("Attempting to get LLM to fix JSON format...")
        fix_llm = LLM(
            model="groq/llama-3.1-8b-instant",
            temperature=0.1  # Lower temperature for more precise output
        )
        
        fix_messages = [
            {"role": "system", "content": "You are a JSON repair expert. Fix the following JSON to make it valid. Return ONLY the valid JSON, no explanations."},
            {"role": "user", "content": f"Fix this JSON: {response[:2000]}"}
        ]
        
        try:
            fixed_response = fix_llm.call(messages=fix_messages)
            json_match = re.search(r'\{.*\}', fixed_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        raise ValueError(f"Could not extract valid JSON from response: {response[:500]}")

    @start()
    def get_user_input(self):
        """Get input from the user about the guide topic and audience"""
        print("\n=== Create Your Comprehensive Guide ===\n")

        # Get user input
        self.state.topic = input("What topic would you like to create a guide for? ")

        # Get audience level with validation
        while True:
            audience = input("Who is your target audience? (beginner/intermediate/advanced) ").lower()
            if audience in ["beginner", "intermediate", "advanced"]:
                self.state.audience_level = audience
                break
            print("Please enter 'beginner', 'intermediate', or 'advanced'")

        print(f"\nCreating a guide on {self.state.topic} for {self.state.audience_level} audience...\n")
        return self.state

    @listen(get_user_input)
    def create_guide_outline(self, state):
        """Create a structured outline for the guide using a direct LLM call"""
        print("Creating guide outline...")

        # Use a more capable model for better JSON output
        llm = LLM(
            model="groq/llama-3.3-70b-versatile",  # More capable model
            temperature=0.3,  # Lower temperature for more consistent output
            max_tokens=2000
        )

        # Simplified prompt with explicit formatting
        messages = [
            {"role": "system", "content": """You are a JSON generator. Your sole purpose is to output valid JSON.
Never include explanatory text, markdown, or anything outside the JSON structure.
Always use double quotes for strings and property names.
Always include commas between properties and array elements.
Never include trailing commas."""},
            {"role": "user", "content": f"""Generate a JSON object for a guide outline about "{state.topic}" for {state.audience_level} level learners.

Required JSON structure:
{{
    "title": "string",
    "introduction": "string", 
    "target_audience": "string",
    "sections": [
        {{"title": "string", "description": "string"}}
    ],
    "conclusion": "string"
}}

Generate 4-6 sections. Return ONLY the JSON object."""}
        ]

        # Make the LLM call with retry logic
        max_retries = 3
        outline_dict = None
        
        for attempt in range(max_retries):
            try:
                response = llm.call(messages=messages)
                
                # Remove markdown if present
                response = re.sub(r'```json\s*', '', response)
                response = re.sub(r'```\s*', '', response)
                response = response.strip()
                
                # Extract and parse JSON
                outline_dict = self.extract_json_from_response(response)
                
                # Validate required fields
                required_fields = ["title", "introduction", "target_audience", "sections", "conclusion"]
                for field in required_fields:
                    if field not in outline_dict:
                        raise ValueError(f"Missing required field: {field}")
                
                # Validate sections
                if not isinstance(outline_dict["sections"], list) or len(outline_dict["sections"]) < 3:
                    raise ValueError(f"Sections should be a list with at least 3 items, got {len(outline_dict.get('sections', []))}")
                
                # Add target_audience if missing
                if "target_audience" not in outline_dict:
                    outline_dict["target_audience"] = state.audience_level
                
                break  # Success, exit retry loop
                
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    # Last attempt failed, create a fallback outline
                    print("Creating fallback outline...")
                    outline_dict = {
                        "title": f"Complete Guide to {state.topic}",
                        "introduction": f"This comprehensive guide will teach you everything about {state.topic}.",
                        "target_audience": state.audience_level,
                        "sections": [
                            {"title": "Getting Started", "description": f"Introduction to {state.topic} basics"},
                            {"title": "Core Concepts", "description": f"Essential {state.topic} concepts for {state.audience_level}s"},
                            {"title": "Best Practices", "description": "Industry best practices and tips"},
                            {"title": "Advanced Topics", "description": "Deep dive into advanced areas"},
                            {"title": "Resources and Next Steps", "description": "Additional resources and learning path"}
                        ],
                        "conclusion": f"Congratulations on completing this guide to {state.topic}!"
                    }
                else:
                    # Add more specific instructions on retry
                    messages[-1]["content"] += "\n\nMake sure your response is a valid JSON object with proper commas between fields."
                    continue

        # Create the guide outline object
        self.state.guide_outline = GuideOutline(**outline_dict)

        # Ensure output directory exists before saving
        os.makedirs("output", exist_ok=True)

        # Save the outline to a file
        with open("output/guide_outline.json", "w") as f:
            json.dump(outline_dict, f, indent=2)

        print(f"✓ Guide outline created with {len(self.state.guide_outline.sections)} sections")
        return self.state.guide_outline

    @listen(create_guide_outline)
    def write_and_compile_guide(self, outline):
        """Write all sections and compile the guide"""
        print("\nWriting guide sections and compiling...")
        completed_sections = []

        # Process sections one by one to maintain context flow
        for idx, section in enumerate(outline.sections, 1):
            print(f"\n[{idx}/{len(outline.sections)}] Processing section: {section.title}")

            # Build context from previous sections
            previous_sections_text = ""
            if completed_sections:
                previous_sections_text = "# Previously Written Sections\n\n"
                for title in completed_sections:
                    previous_sections_text += f"## {title}\n\n"
                    previous_sections_text += self.state.sections_content.get(title, "") + "\n\n"
            else:
                previous_sections_text = "No previous sections written yet."

            # Run the content crew for this section
            try:
                result = ContentCrew().crew().kickoff(inputs={
                    "section_title": section.title,
                    "section_description": section.description,
                    "audience_level": self.state.audience_level,
                    "previous_sections": previous_sections_text,
                    "draft_content": ""
                })
                
                # Store the content
                self.state.sections_content[section.title] = result.raw
                completed_sections.append(section.title)
                print(f"✓ Section completed: {section.title}")
                
            except Exception as e:
                print(f"✗ Error processing section '{section.title}': {e}")
                # Create a fallback section content
                fallback_content = f"""## {section.title}

{section.description}

**Key Points:**
- This section would normally cover important aspects of {self.state.topic}
- Content generation encountered an issue
- Please check your CrewAI configuration

*You can manually edit this section after the guide is generated.*
"""
                self.state.sections_content[section.title] = fallback_content
                completed_sections.append(section.title)

        # Compile the final guide
        print("\nCompiling final guide...")
        guide_content = f"# {outline.title}\n\n"
        guide_content += f"## Introduction\n\n{outline.introduction}\n\n"
        guide_content += f"**Target Audience:** {outline.target_audience}\n\n"
        guide_content += "---\n\n"

        # Add each section in order
        for section in outline.sections:
            section_content = self.state.sections_content.get(section.title, "")
            guide_content += f"{section_content}\n\n"
            guide_content += "---\n\n"

        # Add conclusion
        guide_content += f"## Conclusion\n\n{outline.conclusion}\n\n"

        # Save the guide
        output_file = "output/complete_guide.md"
        with open(output_file, "w") as f:
            f.write(guide_content)

        print(f"\n✓ Complete guide compiled and saved to {output_file}")
        
        # Print summary
        print("\n" + "="*50)
        print("GUIDE CREATION SUMMARY")
        print("="*50)
        print(f"Title: {outline.title}")
        print(f"Topic: {self.state.topic}")
        print(f"Audience: {self.state.audience_level}")
        print(f"Sections: {len(outline.sections)}")
        print(f"Output file: {output_file}")
        print("="*50)
        
        return "Guide creation completed successfully"

def kickoff():
    """Run the guide creator flow"""
    try:
        GuideCreatorFlow().kickoff()
        print("\n=== Flow Complete ===")
        print("Your comprehensive guide is ready in the output directory.")
        print("Open output/complete_guide.md to view it.")
    except KeyboardInterrupt:
        print("\n\nFlow interrupted by user.")
    except Exception as e:
        print(f"\nError running flow: {e}")
        import traceback
        traceback.print_exc()
        raise

def plot():
    """Generate a visualization of the flow"""
    flow = GuideCreatorFlow()
    flow.plot("guide_creator_flow")
    print("Flow visualization saved to guide_creator_flow.html")

if __name__ == "__main__":
    kickoff()