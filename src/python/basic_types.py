#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/1/6 14:18
@Author  : tianshiyang
@File    : basic_types.py
"""
from typing import List, Dict, Tuple, Union, Optional, Literal, Protocol, TypedDict, NotRequired, Callable, TypeVar, \
    Generic

# 1. 基本类型

# 1.1 数组 对象
names: List[str] = ["a", "b"]
scores: Dict[str, int] = {"math": 90}

# 1.2 Tuple
point: Tuple[int, int] = (10, 20)

# 2. 联合类型
# ID = Union[int, str]
ID = int | str

# 3. 可选操作
"""ts
let name?: string
或者
let name: string | null
"""
name: Optional[str] # 或者 name: str | None

# 4.枚举
"""ts
type Consistency = "Strong" | "Bounded";
"""
Consistency: Literal["Strong", "Bounded"] = "Strong"

# 5. 接口与结构约束：TS interface ⇄ Python Protocol / TypedDict
"""ts
interface Retriever {
  getDocs(query: string): string[];
}
"""

class Retriever(Protocol):
    def get_docs(self, query: str) -> list[str]:
        ...

# 6.对象结构
"""ts
type Meta = {
  user_id: string;
  tenant_id: string;
}
"""
class Meta(TypedDict):
    user_id: str
    tenant_id: Optional[str]

meta: Meta = {
    "user_id": "1",
    "tenant_id": "2",
}

# 7.函数类型：Callable ⇄ Function type
"""ts
type FilterFn = (cfg: Config) => Record<string, any>;
"""
FilterFn = Callable[[dict], dict]
def func(fun: Callable[[dict], dict]) -> dict:
    return fun({"aa": 1, "bb": 2})

# 8、泛型系统：高度相似，但 Python 更“底层”
"""ts
class Box<T> {
  value: T;
}
"""
T = TypeVar("T")
class Box(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value