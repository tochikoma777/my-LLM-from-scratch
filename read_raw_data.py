# Read the raw data file and print some statistics about it
# just for test

import re


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])


text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)
print(result)