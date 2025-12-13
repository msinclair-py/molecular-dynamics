"""
Unit tests for logging_config.py module
"""
import logging
import logging.config
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from molecular_simulations.logging_config import configure_logging


class TestConfigureLogging:
    """Test suite for configure_logging function"""
    
    def test_configure_logging_default(self):
        """Test configure_logging with default parameters"""
        # Clear any existing handlers
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        
        configure_logging()
        
        # Should have at least one handler (console)
        assert len(root_logger.handlers) >= 1
        
        # Cleanup
        for handler in root_logger.handlers[:]:
            if handler not in original_handlers:
                root_logger.removeHandler(handler)
    
    def test_configure_logging_with_level(self):
        """Test configure_logging with custom level"""
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        
        configure_logging(level='DEBUG')
        
        # Should have console handler
        assert len(root_logger.handlers) >= 1
        
        # Cleanup
        for handler in root_logger.handlers[:]:
            if handler not in original_handlers:
                root_logger.removeHandler(handler)
    
    def test_configure_logging_with_file(self):
        """Test configure_logging with file output"""
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'test.log'
            
            configure_logging(to_file=str(log_file))
            
            # Should have at least two handlers (console + file)
            assert len(root_logger.handlers) >= 2
        
        # Cleanup
        for handler in root_logger.handlers[:]:
            if handler not in original_handlers:
                root_logger.removeHandler(handler)
                handler.close()
    
    def test_configure_logging_with_env_var(self):
        """Test configure_logging respects environment variables"""
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        
        with patch.dict(os.environ, {'MS_LOG_LEVEL': 'WARNING'}):
            configure_logging()
        
        # Cleanup
        for handler in root_logger.handlers[:]:
            if handler not in original_handlers:
                root_logger.removeHandler(handler)
    
    def test_configure_logging_with_custom_format(self):
        """Test configure_logging with custom format"""
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        
        custom_fmt = "%(levelname)s - %(message)s"
        configure_logging(fmt=custom_fmt)
        
        # Cleanup
        for handler in root_logger.handlers[:]:
            if handler not in original_handlers:
                root_logger.removeHandler(handler)
    
    def test_context_filter_adds_hostname(self):
        """Test that the context filter adds hostname to records"""
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        
        configure_logging()
        
        # Get a test logger and emit a message
        test_logger = logging.getLogger('test_logger')
        
        # Check that we can log without error (the filter runs)
        test_logger.info('Test message')
        
        # Cleanup
        for handler in root_logger.handlers[:]:
            if handler not in original_handlers:
                root_logger.removeHandler(handler)
    
    def test_context_filter_mpi_fallback(self):
        """Test that context filter handles missing mpi4py gracefully"""
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        
        # Should work even without mpi4py
        configure_logging()
        
        test_logger = logging.getLogger('test_logger_mpi')
        test_logger.info('Test MPI fallback')
        
        # Cleanup
        for handler in root_logger.handlers[:]:
            if handler not in original_handlers:
                root_logger.removeHandler(handler)

    def test_configure_logging_with_file_env_var(self):
        """Test configure_logging with MS_LOG_FILE env var"""
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'env_test.log'
            
            with patch.dict(os.environ, {'MS_LOG_FILE': str(log_file)}):
                configure_logging()
            
            # Should have file handler
            assert len(root_logger.handlers) >= 2
        
        # Cleanup
        for handler in root_logger.handlers[:]:
            if handler not in original_handlers:
                root_logger.removeHandler(handler)
                handler.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
