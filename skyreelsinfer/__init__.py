class TaskType:
    T2V = 0
    I2V = 1

class OffloadConfig:
    def __init__(self, high_cpu_memory=False, parameters_level=False, compiler_transformer=False, compiler_cache=""):
        self.high_cpu_memory = high_cpu_memory
        self.parameters_level = parameters_level
        self.compiler_transformer = compiler_transformer
        self.compiler_cache = compiler_cache
