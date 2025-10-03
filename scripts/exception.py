# Can be placed at the top of optimizer.py or in a separate exceptions file

class WorkflowSyntaxError(Exception):
    """
    当工作流出现可自动修复的语法错误（如缺少导入）时抛出。
    """
    def __init__(self, message, original_error=None, round_number=None):
        super().__init__(message)
        self.original_error = original_error
        self.round_number = round_number

    def __str__(self):
        return f"WorkflowSyntaxError in round {self.round_number}: {super().__str__()} (Original: {type(self.original_error).__name__})"
    # Can be placed at the top of optimizer.py or in a separate exceptions file


class WorkflowAttributeError(Exception):
    """当工作流因访问一个被注释掉或不存在的属性而失败时抛出。"""
    def __init__(self, message, original_error=None, round_number=None, filename=None):
        super().__init__(message)
        self.original_error = original_error
        self.round_number = round_number
        self.filename = filename # 记录出错的文件名

    def __str__(self):
        return f"WorkflowAttributeError in round {self.round_number} (file: {self.filename}): {super().__str__()}"
    

    