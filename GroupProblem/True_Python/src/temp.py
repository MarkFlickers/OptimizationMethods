import sys

# -----------------------------------------------------------------------------
# This file was created and refactored with the assistance of ChatGPT (OpenAI).
# Original logic, algorithms and intent were preserved while improving structure,
# readability and adherence to SOLID principles.
#
# The author of the project retains all rights to the original idea, logic and
# specifications. ChatGPT is a tool and does not claim authorship or copyright.
#
# You are free to use, modify and distribute this file as part of your project.
# -----------------------------------------------------------------------------

class PyVersion:
    def __init__(self):
        print(f"Python version: {sys.version}")
        print(f"Version info: {sys.version_info}")

