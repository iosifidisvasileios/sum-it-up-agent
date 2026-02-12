"""
Example usage of the Sum-It-Up Agent.
"""

import asyncio
import warnings
warnings.filterwarnings("ignore")
from sum_it_up_agent.agent import AudioProcessingAgent, AgentConfig


async def _main_with_file(audio_file_path: str):
    """Run example with specific audio file."""
    config = AgentConfig()
    
    agent = AudioProcessingAgent(config)
    async with agent:
        prompt = ("Summarize it and send to my email the action points "
                       "as well as the list of bullet points. "
                       "My email is billiosifidis@gmail.com and bill.iosifidis@hotmail.com. Also extract action points "
                       "and add them to my jira board. Do not use em dash!")


        result = await agent.process_request(audio_file_path, prompt)

    if result.success:
        print(f"✅ Success! Duration: {result.total_duration:.2f}s")
        print(f"📄 Transcription: {result.transcription_file}")
        print(f"📋 Summary: {result.summary_file}")
    else:
        print(f"❌ Failed: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(_main_with_file("/home/vios/Downloads/System-design.mp3"))
