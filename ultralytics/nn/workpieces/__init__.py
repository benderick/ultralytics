from pathlib import Path
import pkgutil
import importlib
import inspect
import yaml
from types import FunctionType

__all__ = []

# 动态导入所有子模块的公开对象
package_dir = __path__
for (_, module_name, _) in pkgutil.iter_modules(package_dir):
    module = importlib.import_module(f".{module_name}", __package__)
    for name in getattr(module, '__all__', []):
        globals()[name] = getattr(module, name)
        __all__.append(name)

# 检查类的 __init__ 方法，分类处理
classes_with_n = []      # 包含参数 n 的类
classes_without_n = []   # 不含参数 n 的类

for name in __all__:
    obj = globals()[name]
    if inspect.isclass(obj):
        init_method = getattr(obj, '__init__', None)
        if init_method and isinstance(init_method, FunctionType):
            try:
                sig = inspect.signature(init_method)
                if 'n' in sig.parameters:
                    classes_with_n.append(name)
                else:
                    classes_without_n.append(name)
            except ValueError:
                print(f"Warning: Could not inspect __init__ of {name}")
        else:
            # 如果没有 __init__ 或不是函数类型，默认归为不含 n
            classes_without_n.append(name)

# YAML 文件路径
# 加载components配置
yaml_file = Path(__file__).parent.parent / 'components.yaml'

# 读取现有的 YAML 文件
try:
    with open(yaml_file, 'r') as f:
        yaml_data = yaml.safe_load(f) or {}
except FileNotFoundError:
    # 如果文件不存在，初始化一个空的结构
    yaml_data = {
        'base_modules': [],
        'repeat_modules': []
    }
except IOError as e:
    print(f"Error reading {yaml_file}: {e}")
    yaml_data = {
        'base_modules': [],
        'repeat_modules': []
    }

# 获取yaml中的 base_modules 和 repeat_modules
current_base_modules = yaml_data.get('base_modules', [])
current_repeat_modules = yaml_data.get('repeat_modules', [])

def is_subset(list1, list2):
    return set(list1).issubset(set(list2))

# 检查是否需要更新yaml
if is_subset(classes_with_n, current_base_modules) and is_subset(classes_without_n, current_base_modules) and is_subset(classes_with_n, current_repeat_modules):
    pass
else:
    # base_modules 包含所有类（去重）
    updated_base_modules = list(set(current_base_modules + classes_with_n + classes_without_n))
    # repeat_modules 只包含带 n 的类（去重）
    updated_repeat_modules = list(set(current_repeat_modules + classes_with_n))


    yaml_data['base_modules'] = updated_base_modules
    yaml_data['repeat_modules'] = updated_repeat_modules

    # 写入更新后的 YAML 文件
    try:
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
        print(f"🔔 WORKPIECES successfully updated components.yaml")
    except IOError as e:
        print(f"WORKPIECES Error writing to {yaml_file}: {e}")