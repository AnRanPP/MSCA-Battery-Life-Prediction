# models/__init__.py
# 只导入存在的模型，避免导入错误

__all__ = []

try:
    from .PCE import Model as PCE_Model
    __all__.append('PCE_Model')
except ImportError:
    pass

try:
    from .CPMLP import Model as CPMLP_Model
    __all__.append('CPMLP_Model')
except ImportError:
    pass

# 其他模型可以类似添加
