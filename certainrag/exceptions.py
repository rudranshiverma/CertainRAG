class CertainRAGError(Exception):
    """Base exception for CertainRAG."""
    pass
class InputValidationError(CertainRAGError):
    """Raised when input data is invalid."""
    pass
class ModelLoadError(CertainRAGError):
    """Raised when the required model cannot be loaded."""
    pass
class ComputationError(CertainRAGError):
    """Raised when uncertainty computation fails."""
    pass