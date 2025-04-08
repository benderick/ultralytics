# 检测头
from .AFPNHead import *
from .AFPN4Head import *
from .ASFFHead import *
from .ASFF4Head import *

# 注意力
from .TripleAttention import *

# 上采样
from .Dysample import *

detection_head_list = (AFPNHead, AFPN4Head, ASFFHead, ASFF4Head)