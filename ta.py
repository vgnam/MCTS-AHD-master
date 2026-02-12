import re

input_path = r"C:\Users\user\Downloads\network_knowledge.md"
output_path = r"C:\Users\user\Downloads\network_knowledge_clean.md"

with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

# Xóa tất cả dòng trống (kể cả dòng chỉ có space/tab)
text = re.sub(r'^[ \t]*\n', '', text, flags=re.MULTILINE)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Done! Saved to network_knowledge_clean.md")
