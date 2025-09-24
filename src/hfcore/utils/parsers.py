import re
import json

def parse_holding(text: str) -> float | None:
    """
    从模型返回文本中解析 {"holding_tp1": <float>}
    """
    # 先尝试 json
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "holding_tp1" in obj:
            return float(obj["holding_tp1"])
    except Exception:
        pass

    # 再尝试正则
    m = re.search(r'"holding_tp1"\s*:\s*([0-9.eE+-]+)', text)
    if m:
        return float(m.group(1))

    return None