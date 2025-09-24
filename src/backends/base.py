from abc import ABC, abstractmethod

__all__ = ["LLMBackend"]

class LLMBackend(ABC):
    """统一接口：不同模型后端的基类"""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        给定 prompt，返回模型的字符串输出
        子类必须实现
        """
        raise NotImplementedError("Subclasses must implement `generate`")