from pathlib import Path

from app.services.ingestion import ManualIngestionService


def test_parse_one_file():
    try:
        import langchain_text_splitters  # noqa: F401
    except ModuleNotFoundError:
        print("[SKIP] 未安装 langchain-text-splitters，跳过 ingestion 切块测试。")
        return

    ingestion_service = ManualIngestionService()
    chunks = ingestion_service.parse_one_file(Path("手册/冰箱手册.txt"))
    assert len(chunks) > 0
    assert chunks[0].text is not None
    assert chunks[0].image_ids is not None
    assert chunks[0].chunk_id is not None
    assert chunks[0].manual_name is not None
    print(chunks)

if __name__ == "__main__":
    test_parse_one_file()


