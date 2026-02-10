#!/usr/bin/env python3
"""
Main application entry point for the sum-it-up-agent.
Singleton that boots MCP servers and provides interactive interface.
"""

import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import dotenv
dotenv.load_dotenv()

from sum_it_up_agent.agent.orchestrator import AudioProcessingAgent
from sum_it_up_agent.agent.models import AgentConfig


class SumItUpApp:
    """Singleton application that manages MCP servers and orchestrates processing."""
    
    _instance: Optional['SumItUpApp'] = None
    _agent: Optional[AudioProcessingAgent] = None
    _server_processes: Dict[str, subprocess.Popen] = {}
    
    def __new__(cls) -> 'SumItUpApp':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._agent is None:
            self._agent = AudioProcessingAgent()
    
    async def boot_mcp_servers(self, wait_for_ready: bool = True, timeout: int = 120):
        """Boot all MCP servers as subprocesses."""
        try:
            print("📡 Booting MCP servers...")
            
            # Server configurations from environment variables
            servers = {
                "audio_processor": {
                    "module": "sum_it_up_agent.audio_processor.mcp_server_audio",
                    "env_vars": {
                        "MCP_TRANSPORT_AUDIO_PROCESSOR": os.getenv("MCP_TRANSPORT_AUDIO_PROCESSOR", "http"),
                        "MCP_HOST_AUDIO_PROCESSOR": os.getenv("MCP_HOST_AUDIO_PROCESSOR", "127.0.0.1"), 
                        "MCP_PORT_AUDIO_PROCESSOR": os.getenv("MCP_PORT_AUDIO_PROCESSOR", "9001"),
                        "MCP_PATH_AUDIO_PROCESSOR": os.getenv("MCP_PATH_AUDIO_PROCESSOR", "/audio_proc")
                    }
                },
                "topic_classifier": {
                    "module": "sum_it_up_agent.topic_classification.mcp_topic_classification",
                    "env_vars": {
                        "MCP_TRANSPORT_TOPIC_CLASSIFIER": os.getenv("MCP_TRANSPORT_TOPIC_CLASSIFIER", "http"),
                        "MCP_HOST_TOPIC_CLASSIFIER": os.getenv("MCP_HOST_TOPIC_CLASSIFIER", "127.0.0.1"),
                        "MCP_PORT_TOPIC_CLASSIFIER": os.getenv("MCP_PORT_TOPIC_CLASSIFIER", "9002"),  # From .env.example
                        "MCP_PATH_TOPIC_CLASSIFIER": os.getenv("MCP_PATH_TOPIC_CLASSIFIER", "/classifier")  # From .env.example
                    }
                },
                "summarizer": {
                    "module": "sum_it_up_agent.summarizer.mcp_summarizer",
                    "env_vars": {
                        "MCP_TRANSPORT_SUMMARIZER": os.getenv("MCP_TRANSPORT_SUMMARIZER", "http"),
                        "MCP_HOST_SUMMARIZER": os.getenv("MCP_HOST_SUMMARIZER", "127.0.0.1"),
                        "MCP_PORT_SUMMARIZER": os.getenv("MCP_PORT_SUMMARIZER", "9000"),  # From .env.example
                        "MCP_PATH_SUMMARIZER": os.getenv("MCP_PATH_SUMMARIZER", "/summarizer")  # From .env.example
                    }
                },
                "communicator": {
                    "module": "sum_it_up_agent.communicator.mcp_communicator",
                    "env_vars": {
                        "MCP_TRANSPORT_COMMUNICATOR": os.getenv("MCP_TRANSPORT_COMMUNICATOR", "http"),
                        "MCP_HOST_COMMUNICATOR": os.getenv("MCP_HOST_COMMUNICATOR", "127.0.0.1"),
                        "MCP_PORT_COMMUNICATOR": os.getenv("MCP_PORT_COMMUNICATOR", "9003"),
                        "MCP_PATH_COMMUNICATOR": os.getenv("MCP_PATH_COMMUNICATOR", "/communicate")
                    }
                }
            }
            
            # Start each server
            for server_name, config in servers.items():
                await self._start_server(server_name, config["module"], config["env_vars"])
            
            if wait_for_ready:
                await self._wait_for_servers_ready(timeout)
                
            print("✅ All servers ready!")
            
        except Exception as e:
            print(f"❌ Failed to boot MCP servers: {e}")
            await self.stop_mcp_servers()
            raise
    
    async def _start_server(self, name: str, module: str, env_vars: Dict[str, str]):
        """Start a single MCP server with clean environment."""
        if name in self._server_processes:
            print(f"⚠️  Server {name} is already running")
            return
            
        print(f"🚀 Starting {name} server...")
        
        # Prepare clean environment - only load what we need
        env = {}
        
        # Copy essential system variables
        essential_vars = ['PATH', 'HOME', 'USER', 'SHELL', 'TERM', 'LANG', 'PYTHONPATH']
        for var in essential_vars:
            if var in os.environ:
                env[var] = os.environ[var]
        
        # Add server-specific environment variables
        env.update(env_vars)
        
        # Add only essential environment variables that servers actually need
        if name == "audio_processor":
            # Audio processor only needs HuggingFace token
            if 'HUGGINGFACE_TOKEN' in os.environ:
                env['HUGGINGFACE_TOKEN'] = os.environ['HUGGINGFACE_TOKEN']
                
        elif name == "topic_classifier":
            # Topic classifier needs HuggingFace token
            if 'HUGGINGFACE_TOKEN' in os.environ:
                env['HUGGINGFACE_TOKEN'] = os.environ['HUGGINGFACE_TOKEN']
                
        elif name == "summarizer":
            # Summarizer needs LLM provider settings
            llm_vars = ['OPENAI_API_KEY', 'OLLAMA_HOST', 'OPENAI_BASE_URL', 'ANTHROPIC_API_KEY']
            for var in llm_vars:
                if var in os.environ:
                    env[var] = os.environ[var]
                    
        elif name == "communicator":
            # Communicator needs email settings
            email_vars = ['EMAIL_SMTP_HOST', 'EMAIL_SMTP_PORT', 'EMAIL_USERNAME', 'EMAIL_PASSWORD', 'EMAIL_FROM']
            for var in email_vars:
                if var in os.environ:
                    env[var] = os.environ[var]
        
        # Start the server process with clean environment
        process = subprocess.Popen(
            [sys.executable, "-m", module],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self._server_processes[name] = process
        print(f"✅ Started {name} server with PID {process.pid}")
    
    async def _wait_for_servers_ready(self, timeout: int = 120):
        """Wait for all servers to be ready."""
        print("⏳ Waiting for servers to be ready...")
        print("⏱️  Giving servers 5 seconds to initialize...")
        await asyncio.sleep(15)  # Give servers more time to start
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            all_ready = True
            
            for name, process in self._server_processes.items():
                if process.poll() is not None:
                    # Process has terminated
                    stdout, stderr = process.communicate()
                    raise RuntimeError(f"Server {name} failed to start: {stderr}")
                
                # Simple health check - just check if port is open
                try:
                    import socket
                    if name == "audio_processor":
                        host = os.getenv("MCP_HOST_AUDIO_PROCESSOR", "127.0.0.1")
                        port = int(os.getenv("MCP_PORT_AUDIO_PROCESSOR", "9001"))
                    elif name == "topic_classifier":
                        host = os.getenv("MCP_HOST_TOPIC_CLASSIFIER", "127.0.0.1")
                        port = int(os.getenv("MCP_PORT_TOPIC_CLASSIFIER", "9002"))
                    elif name == "summarizer":
                        host = os.getenv("MCP_HOST_SUMMARIZER", "127.0.0.1")
                        port = int(os.getenv("MCP_PORT_SUMMARIZER", "9000"))
                    elif name == "communicator":
                        host = os.getenv("MCP_HOST_COMMUNICATOR", "127.0.0.1")
                        port = int(os.getenv("MCP_PORT_COMMUNICATOR", "9003"))
                    else:
                        continue
                        
                    print(f"🔍 Checking {name} at {host}:{port}...")
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2)
                    result = sock.connect_ex((host, port))
                    sock.close()
                    
                    if result == 0:
                        print(f"📊 {name} port is open (ready)")
                    else:
                        all_ready = False
                        print(f"⚠️  {name} port not accessible (code {result})")
                        
                except Exception as e:
                    all_ready = False
                    print(f"⚠️  {name} health check failed: {e}")
                    break
            
            if all_ready:
                print("✅ All servers are ready")
                return
                
            await asyncio.sleep(3)  # Wait longer between checks
        
        raise TimeoutError(f"Servers did not become ready within {timeout} seconds")
    
    async def stop_mcp_servers(self):
        """Stop all MCP servers."""
        print("🛑 Stopping MCP servers...")
        
        for name, process in self._server_processes.items():
            try:
                if process.poll() is None:  # Process is still running
                    print(f"🔄 Stopping {name} server (PID {process.pid})...")
                    process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print(f"⚡ Force killing {name} server...")
                        process.kill()
                        process.wait()
                        
            except Exception as e:
                print(f"❌ Error stopping {name} server: {e}")
        
        self._server_processes.clear()
        print("✅ All MCP servers stopped")
    
    async def start(self):
        """Start the application - boot servers and prepare for processing."""
        print("🚀 Starting Sum-It-Up Agent...")
        print("=" * 50)
        
        try:
            # Boot MCP servers
            await self.boot_mcp_servers()
            
            # Initialize clients
            print("🔗 Initializing clients...")
            await self._agent.initialize()
            print("✅ Ready to process!")
            print("=" * 50)
            
        except Exception as e:
            print(f"❌ Failed to start: {e}")
            await self.shutdown()
            sys.exit(1)
    
    async def run_interactive(self):
        """Run interactive mode with user input."""
        print("🎯 Sum-It-Up Agent - Interactive Mode")
        print("Type 'quit' or 'exit' to stop")
        print("=" * 50)
        
        while True:
            try:
                # Get audio file path
                audio_file = input("\n📁 Enter audio file path: ").strip()
                
                if audio_file.lower() in ['quit', 'exit']:
                    break
                
                if not audio_file:
                    print("❌ Please enter a valid file path")
                    continue
                
                # Validate file exists
                if not Path(audio_file).exists():
                    print(f"❌ File not found: {audio_file}")
                    continue
                
                # Get user prompt
                user_prompt = input("💬 What would you like to do with this audio? ").strip()
                
                if user_prompt.lower() in ['quit', 'exit']:
                    break
                
                if not user_prompt:
                    print("❌ Please enter a description of what you want to do")
                    continue
                
                # Process the request
                print(f"\n⚡ Processing {audio_file}...")
                print(f"📝 Request: {user_prompt}")
                print("-" * 30)
                
                result = await self._agent.process_request(audio_file, user_prompt)
                
                # Display results
                if result.success:
                    print("✅ Processing completed successfully!")
                    
                    if result.transcription_file:
                        print(f"📄 Transcription: {result.transcription_file}")
                    
                    if result.summary_file:
                        print(f"📋 Summary: {result.summary_file}")
                    
                    if result.communication_results:
                        print("📧 Communication sent:")
                        for comm in result.communication_results:
                            status = "✅" if comm.get('success', False) else "❌"
                            print(f"  {status} {comm.get('channel', 'unknown')}")
                    
                    print(f"⏱️  Total time: {result.total_duration:.2f}s")
                else:
                    print(f"❌ Processing failed: {result.error_message}")
                
                print("=" * 50)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
    
    async def process_once(self, audio_file: str, user_prompt: str):
        """Process a single request and return result."""
        try:
            result = await self._agent.process_request(audio_file, user_prompt)
            return result
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            return None
    
    async def shutdown(self):
        """Shutdown the application gracefully."""
        print("\n🛑 Shutting down...")
        
        # Stop MCP servers
        await self.stop_mcp_servers()
        
        if self._agent:
            await self._agent.cleanup()
        
        print("👋 Goodbye!")


# Global app instance
_app: Optional[SumItUpApp] = None


def get_app() -> SumItUpApp:
    """Get the singleton app instance."""
    global _app
    if _app is None:
        _app = SumItUpApp()
    return _app


async def main():
    """Main entry point."""
    app = get_app()
    
    try:
        await app.start()
        
        if len(sys.argv) > 1:
            # Command line mode
            if len(sys.argv) >= 3:
                audio_file = sys.argv[1]
                user_prompt = " ".join(sys.argv[2:])
                
                print(f"📁 File: {audio_file}")
                print(f"💬 Request: {user_prompt}")
                print("-" * 30)
                
                result = await app.process_once(audio_file, user_prompt)
                
                if result and result.success:
                    print("✅ Completed successfully!")
                    if result.transcription_file:
                        print(f"📄 Transcription: {result.transcription_file}")
                    if result.summary_file:
                        print(f"📋 Summary: {result.summary_file}")
                else:
                    print("❌ Failed")
            else:
                print("Usage: python -m sum_it_up_agent.app <audio_file> <user_prompt>")
        else:
            # Interactive mode
            await app.run_interactive()
            
    except KeyboardInterrupt:
        print("\n👋 Interrupted")
    finally:
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
