"""
Centralized logging configuration for the agent module.
"""

import logging
from typing import Optional

from sum_it_up_agent.observability.logger import get_logger as _get_logger

def get_agent_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get the unified agent logger.
    
    Args:
        name: Optional name for the logger (e.g., 'prompt_parser', 'orchestrator')
        
    Returns:
        Logger instance
    """
    if name:
        return _get_logger(f"sum_it_up_agent.agent.{name}")
    return _get_logger("sum_it_up_agent.agent")
