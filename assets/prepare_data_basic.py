from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
import re


def preprocess_text_OCR(images):
    pages = []
    
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img, lang="vie")
        lines = [line.strip() for line in text.splitlines() if line.strip() != ""]

        if i == 0:
            lines = lines[:-2] if len(lines) >= 2 else []
        else:
            lines = lines[2:-2] if len(lines) >= 4 else []

        processed_text = "\n".join(lines)
        pages.append(processed_text)
    
    return pages


def preprocess_text_PDF(reader):
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if not text:
            pages.append("")
            continue

        lines = [line.strip() for line in text.splitlines() if line.strip() != ""]
        if len(lines) >= 2:
            lines = lines[:-2]
        else:
            lines = []

        pages.append("\n".join(lines))

    return pages



def text_to_markdown(text):
    lines = text.strip().splitlines()
    result = []
    i = 0

    found_title = False

    while i < len(lines):
        word = lines[i].strip()

        if not word or re.fullmatch(r"-{3,}", word):
            i += 1
            continue

        if not found_title and word == "LUẬT" and i + 1 < len(lines):
            title = lines[i + 1].strip()
            result.append(f"# LUẬT: {title}")
            found_title = True
            i += 2
            continue

        if re.match(r"^Chương\s+[IVXLC\d]+", word):
            chapter_line = word
            chapter_title = lines[i + 1].strip() if i + 1 < len(lines) else ""
            result.append(f"## {chapter_line} - {chapter_title}")
            i += 2
            continue

        if re.match(r"^Điều\s+\d+", word):
            result.append(f"### {word}")
            i += 1
            continue

        result.append(word)
        i += 1

    return "\n".join(result)


def save_pages_to_txt(filename, text):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)


nq_path = "/content/nq_193_2025_QH15-tool.pdf"
luat_path = "/content/L_64_2025_QH15.pdf"

images = convert_from_path(nq_path, dpi=300)
reader = PdfReader(luat_path)

nghi_quyet_pages = preprocess_text_OCR(images)
nghi_quyet_text = "\n".join(nghi_quyet_pages)
luat_pages = preprocess_text_PDF(reader)
luat_text = "\n".join(luat_pages)
markdown_luat = text_to_markdown(luat_text)

save_pages_to_txt("nq_193_2025_QH15_cleaned.txt", nghi_quyet_text)
save_pages_to_txt("L_64_2025_QH15_cleaned.txt", markdown_luat)



    def retrieve(self, query: Union[str, List[str]]) -> str:
        if isinstance(query, str):
            queries = [query]
        elif isinstance(query, list):
            queries = query
        else:
            raise ValueError("Query must be a string or list of strings")
        
        def write_log(message: str, filepath: str = "D:/legal_rag_project/debug_retriever_log.txt"):
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(f"{message}\n\n")


        write_log(f"[INPUT QUERY]\n{queries}")

        # Vector retrieval
        vector_docs_with_scores = self.vector_retriever.get_unique_parent_docs_with_scores(queries)
        vector_log = "\n\n".join(
            f"[Vector] Score: {score:.4f} | ID: {doc.metadata.get('parent_id')} | Content: {doc.page_content}"
            for doc, score in vector_docs_with_scores
        )
        write_log(f"[VECTOR RETRIEVER]\n{vector_log}")

        # Keyword retrieval
        use_512 = (self.chunk_size == 512)
        keyword_docs_with_scores = self.keyword_retriever.get_unique_parent_docs_with_scores(queries, use_512=use_512)
        keyword_log = "\n\n".join(
            f"[Keyword] Score: {score:.4f} | ID: {doc.metadata.get('parent_id')} | Content: {doc.page_content}"
            for doc, score in keyword_docs_with_scores
        )
        write_log(f"[KEYWORD RETRIEVER]\n{keyword_log}")

        # Hybrid merging
        merged_docs = self._weighted_merge_and_select_docs(vector_docs_with_scores, keyword_docs_with_scores)
        merged_log = "\n\n".join(
            f"[Hybrid] ID: {doc.metadata.get('parent_id')} | Content: {doc.page_content}"
            for doc in merged_docs
        )
        write_log(f"[HYBRID MERGE RESULT]\n{merged_log}")

        # Reranking
        reranked_docs_with_scores = self.reranker.rerank_documents(queries, merged_docs)
        rerank_log = "\n\n".join(
            f"[Reranker] Final Score: {score:.4f} | ID: {doc.metadata.get('parent_id')} | Content: {doc.page_content}"
            for doc, score in reranked_docs_with_scores
        )
        write_log(f"[RERANKER RESULTS]\n{rerank_log}")

        # Final output
        reranked_docs = [doc for doc, _ in reranked_docs_with_scores]
        final_docs = reranked_docs[:self.num_final_docs]
        final_log = "\n\n".join(
            f"[FINAL SELECTED DOC] ID: {doc.metadata.get('parent_id')} | Content: {doc.page_content}"
            for doc in final_docs
        )
        write_log(f"[FINAL OUTPUT]\n{final_log}")

        return self._format_context(final_docs)
