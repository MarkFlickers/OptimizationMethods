import sys
import time

start_time = time.time()

for i in range(1000000):
    pass 

print(f"Python version: {sys.version}")
print(f"Version info: {sys.version_info}")

end_time = time.time()
execution_time_check = end_time - start_time
print(f"{execution_time_check:.4f}")
pass