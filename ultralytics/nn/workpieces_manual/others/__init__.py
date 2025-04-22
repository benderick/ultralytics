import pkgutil
import importlib

# 应该手动导入的，因为它们需在task.py中明确使用，自动也行，不过会导致代码自动检测爆黄

from .Dysample import *
from .EIEStem import *
from .LAWDS import *
from .SBA import *
from .TripleAttention import *
from .CSPOmniKernel import *
from .SimAM import *
from .BlurPool import *
from .D2SUpsample import *

# __all__ = []

# # 动态导入所有子模块的公开对象
# package_dir = __path__
# for (_, module_name, _) in pkgutil.iter_modules(package_dir):
#     module = importlib.import_module(f".{module_name}", __package__)
#     for name in getattr(module, '__all__', []):
#         globals()[name] = getattr(module, name)
#         __all__.append(name)