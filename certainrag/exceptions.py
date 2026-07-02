class CertainRAGError(Exception):
    pass
class ConfigurationError(CertainRAGError):
    pass
class MissingDependencyError(CertainRAGError):
    pass
class BackendError(CertainRAGError):
    pass