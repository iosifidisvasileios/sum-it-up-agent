"""
Centralized logging configuration for the agent module.
"""

import logging
from typing import Optional


class AgentLogger:
    """Unified logger for the agent module."""
    
    _instance: Optional['AgentLogger'] = None
    _logger: Optional[logging.Logger] = None
    
    def __new__(cls) -> 'AgentLogger':
        """Singleton pattern to ensure one logger instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the unified logger."""
        if self._logger is None:
            self._logger = self._setup_logging()
    
    @staticmethod
    def _setup_logging() -> logging.Logger:
        """Setup logging configuration for the agent module."""
        logger = logging.getLogger("sum_it_up_agent.agent")
        
        # Only setup handlers if they haven't been configured yet
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            
            # Prevent propagation to root logger to avoid duplicate logs
            logger.propagate = False
        
        return logger
    
    @property
    def logger(self) -> logging.Logger:
        """Get the logger instance."""
        return self._logger
    
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """
        Get a logger with a specific name or the default agent logger.
        
        Args:
            name: Optional name for the logger (e.g., 'prompt_parser', 'orchestrator')
            
        Returns:
            Logger instance
        """
        if name:
            return logging.getLogger(f"sum_it_up_agent.agent.{name}")
        return self._logger


# Global instance for easy access
_agent_logger = AgentLogger()

def get_agent_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get the unified agent logger.
    
    Args:
        name: Optional name for the logger (e.g., 'prompt_parser', 'orchestrator')
        
    Returns:
        Logger instance
    """
    return _agent_logger.get_logger(name)
