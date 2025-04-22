# 检测头
from .head import *

# 其它，需要自定义参数的
from .others import *


# 检测头列表---------------------------------
detection_head_list = []

# 检查 head 模块是否有 __all__ 属性
if hasattr(head, '__all__'):
    # 遍历 __all__ 中定义的名称
    for name in head.__all__:
        try:
            # 获取名称对应的对象并添加到列表中
            detection_head_list.append(getattr(head, name))
        except AttributeError:
            # 如果 __all__ 中的名称在模块中实际不存在，则跳过
            print(f"警告：head.__all__ 中定义的 '{name}' 在 head 模块中未找到。")
else:
    print("警告：head 模块未定义 __all__，将尝试导入所有公共成员。")
