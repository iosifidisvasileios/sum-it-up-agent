#!/usr/bin/env python3
"""
Simple script to run the Sum-It-Up Agent app.
"""

import asyncio
from src.sum_it_up_agent.app import get_app

async def main():
    """Run the app."""
    app = get_app()
    
    try:
        await app.start()
        await app.run_interactive()
    finally:
        await app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
