from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
# 专为你这10份多品类手册定制的混合正则切分符
custom_separators = [
    r"\n+#\s+",                         # 匹配带有一个或多个换行的标题
    r"\n+\d+[\.）\)]?\s+",              # 匹配正常步骤
    r"\n+[a-zA-Z][\.）\)]\s+",          # 匹配字母步骤
    r"\n+[·\-\u00b7\u25cf\u2022]\s+",   # 匹配带换行的无序列表
    r"\s+[·\u00b7\u25cf\u2022]\s+",     # 【新增】专门抓取没换行的内联无序列表 (解决瑕疵1)
    r"\n\n",
    r"\n",
    r" ",
    r""
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,        # 软约束：尽量保持在这个长度以内
    chunk_overlap=120,     # 重叠度稍微放大，防止上下文断裂
    separators=custom_separators,
    is_separator_regex=True,
    keep_separator=True    # 【关键参数】保留切分符，确保丢给大模型时，步骤序号 "1. " 还在！
)
text = Path("/Users/nikonzhang/compeletion/手册/吹风机手册.txt").read_text()

chunks = text_splitter.split_text(text)
lengths = [len(c) for c in chunks]
for chunk in chunks:
    print(chunk)
    print("-"*100)
print(f"总 Chunk 数: {len(chunks)}")
print(f"平均长度: {sum(lengths)/len(lengths):.0f}")
print(f"最短 Chunk 长度: {min(lengths)}")
print(f"最长 Chunk 长度: {max(lengths)}")