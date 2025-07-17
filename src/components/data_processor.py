import json
from tqdm import tqdm
import re

def load_content_doc(jsonl_path):
    all_doc = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            content = doc.get("content")
            if content:
                all_doc.append(content)
    return all_doc


def preprocess_text(docs):
    processed_docs = []

    DOC_TYPES = [
        "LUẬT", "NGHỊ ĐỊNH", "NGHỊ QUYẾT", "QUYẾT NGHỊ", "QUYẾT ĐỊNH",
        "THÔNG TƯ", "THÔNG TƯ LIÊN TỊCH", "PHÁP LỆNH", "LỆNH", "CHỈ THỊ",
        "CÔNG VĂN", "BIÊN BẢN", "HỢP ĐỒNG", "QUY CHẾ", "ĐIỀU LỆ", "THÔNG BÁO",
        "BÁO CÁO", "KẾ HOẠCH", "PHƯƠNG ÁN", "ĐỀ ÁN", "PHỤ LỤC", "DANH MỤC"
    ]

    LEADING_KEYWORDS = {
        "Căn cứ", "Theo đề nghị", "Chiểu theo", "Xét", "Xét đề nghị"
    }

    def is_full_upper(text):
        return text.isupper() and any(c.isalpha() for c in text)

    def normalize_doc_type(line):
        """
        Trả về DOC_TYPE nếu dòng bắt đầu bằng một DOC_TYPE hợp lệ.
        Với riêng 'LUẬT', yêu cầu cả dòng phải viết hoa toàn bộ (tránh nhầm 'Luật số').
        Các DOC_TYPE khác chỉ cần bắt đầu bằng từ đó (in hoa).
        """
        stripped_line = line.strip()

        for dt in DOC_TYPES:
            if dt == "LUẬT":
                if stripped_line == "LUẬT":
                    return "LUẬT"
            else:
                if stripped_line.upper().startswith(dt):
                    return dt

        return None

    def is_likely_leading_keyword(line):
        return any(line.startswith(k) for k in LEADING_KEYWORDS)

    def is_likely_structure_line(line):
        line = line.strip()
        return (
            re.match(r"^Điều\s+\d+", line) or
            re.match(r"^Chương\s+[IVXLC\d]+", line) or
            re.match(r"^Mục\s+[IVXLC\d]+", line) or
            re.match(r"^[IVXLC]+\.", line) or
            re.match(r"^\d+(\.\d+)*", line)
        )

    def is_separator(line):
        return re.fullmatch(r"[-*]{3,}", line.strip()) is not None

    for doc in tqdm(docs, desc="Processing documents"):
        lines = [line.strip() for line in doc.strip().splitlines()]
        result = []
        doc_type_count = 0
        stage = "search_doc_type"

        # === STEP 1: Tách theo 3 phần như yêu cầu ===
        sep_indices = [i for i, line in enumerate(lines) if is_separator(line)]

        part1, part2, part3 = [], [], []
        doc_type_start_idx = None

        if len(sep_indices) >= 2:
            i1, i2 = sep_indices[0], sep_indices[1]

            # part 1: trước dòng phân cách đầu tiên
            part1 = [l for l in lines[:i1] if l and not is_separator(l)]
            if part1:
                result.append(" ".join(part1))

            # part 2: giữa 2 dòng phân cách
            part2 = [l for l in lines[i1+1:i2] if l and not is_separator(l)]
            if part2:
                result.append(" ".join(part2))

            # part 3: sau phân cách thứ 2 đến trước dòng có DOC_TYPE
            for i, line in enumerate(lines[i2+1:], start=i2+1):
                if is_separator(line):
                    continue
                if normalize_doc_type(line.replace(":", "")):
                    doc_type_start_idx = i
                    break
                part3.append(line)

            if part3:
                result.append(" ".join(part3))

            lines = lines[doc_type_start_idx:] if doc_type_start_idx is not None else []
        else:
            # Nếu không đủ 2 separator, fallback: loại separator và gộp toàn bộ
            all_content = [l for l in lines if l and not is_separator(l)]
            if all_content:
                result.append(" ".join(all_content))
            lines = []  # không xử lý thêm

        # === STEP 2: Xử lý tiêu đề văn bản và phần căn cứ ===
        buffer = []
        i = 0
        while i < len(lines):
            raw_line = lines[i]
            line = raw_line.strip()
            if not line or is_separator(line):
                i += 1
                continue

            normalized = normalize_doc_type(line.replace(":", "").strip())

            if normalized:
                prev_line = lines[i - 1].strip() if i > 0 else ""
                next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""

                is_continuation = is_likely_leading_keyword(prev_line)
                is_valid_doc_type_2 = is_likely_structure_line(next_line)

                if doc_type_count >= 1 and is_continuation:
                    buffer.append(line)
                    i += 1
                    continue

                if doc_type_count >= 1 and not is_valid_doc_type_2:
                    buffer.append(line)
                    i += 1
                    continue

                doc_type_count += 1

                if doc_type_count == 1 and is_full_upper(line):
                    result.append(line)
                    stage = "collect_upper"
                    buffer = []
                    i += 1
                    continue

                if doc_type_count == 2:
                    if buffer:
                        result.append(" ".join(buffer))
                        buffer = []
                    result.append(normalized)
                    i += 1

                    while i < len(lines):
                        tail_line = lines[i].strip()
                        if tail_line and not is_separator(tail_line):
                            result.append(tail_line)
                        i += 1
                    break

            if doc_type_count == 1:
                if stage == "collect_upper":
                    if is_full_upper(line):
                        buffer.append(line)
                        i += 1
                        continue
                    else:
                        if buffer:
                            result.append(" ".join(buffer))
                            buffer = []
                        stage = "collect_lower"
                        continue

                elif stage == "collect_lower":
                    buffer.append(line)
                    i += 1
                    continue

            result.append(line)
            i += 1

        if buffer:
            result.extend(buffer)

        processed_docs.append("\n".join(result))

    return processed_docs


def preprocess_legal_documents_to_markdown(docs):
    processed_docs = []

    DOC_TYPES = [
        "LUẬT", "NGHỊ ĐỊNH", "NGHỊ QUYẾT", "QUYẾT NGHỊ", "QUYẾT ĐỊNH",
        "THÔNG TƯ", "THÔNG TƯ LIÊN TỊCH", "PHÁP LỆNH", "LỆNH", "CHỈ THỊ",
        "CÔNG VĂN", "BIÊN BẢN", "HỢP ĐỒNG", "QUY CHẾ", "ĐIỀU LỆ",
        "THÔNG BÁO", "BÁO CÁO", "KẾ HOẠCH", "PHƯƠNG ÁN", "ĐỀ ÁN",
        "PHỤ LỤC", "DANH MỤC"
    ]

    for doc in docs:
        lines = doc.strip().splitlines()
        result = []

        i = 0
        doc_type_count = 0

        # === Biến trạng thái ===
        inside_dieu = False
        last_number_heading_allowed = False

        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            matched_doc_type = None
            if line == line.upper():
                for dt in DOC_TYPES:
                    if line.startswith(dt):
                        matched_doc_type = dt
                        break

            if matched_doc_type:
                doc_type_count += 1
                heading_level = "#" if doc_type_count == 1 else "##"
                result.append(f"{heading_level} {matched_doc_type.title()}")
                inside_dieu = False
                last_number_heading_allowed = False
                i += 1
                continue

            # 1. Chương
            m = re.match(r"^(Chương\s+[IVXLC\d]+)", line, re.IGNORECASE)
            if m:
                result.append(f"### {m.group(1)}")
                rest = line[m.end():].strip()
                if rest:
                    result.append(rest)
                inside_dieu = False
                last_number_heading_allowed = False
                i += 1
                continue

            # 2. Mục
            m = re.match(r"^(Mục\s+[IVXLC\d]+)", line, re.IGNORECASE)
            if m:
                result.append(f"#### {m.group(1)}")
                rest = line[m.end():].strip()
                if rest:
                    result.append(rest)
                inside_dieu = False
                last_number_heading_allowed = False
                i += 1
                continue

            # 3. Điều
            m = re.match(r"^(Điều\s+\d+\.)", line)
            if m:
                result.append(f"##### {m.group(1)}")
                rest = line[m.end():].strip()
                if rest:
                    result.append(rest)
                inside_dieu = True  # Đang trong một điều
                last_number_heading_allowed = False
                i += 1
                continue

            # 4. I., II., III.
            m = re.match(r"^([IVXLC]+)\.\s*(.*)", line)
            if m:
                result.append(f"### {m.group(1)}.")
                if m.group(2):
                    result.append(m.group(2))
                inside_dieu = False
                last_number_heading_allowed = False
                i += 1
                continue

            # 5. Dạng "1." không kèm nội dung
            if re.match(r"^(?!202\d)([1-9][0-9]{0,2})\.\s*$", line):
                if not inside_dieu:
                    result.append(f"#### {line.strip()}")
                    last_number_heading_allowed = True
                else:
                    result.append(line)  # không markdown nếu trong Điều
                    last_number_heading_allowed = False
                i += 1
                continue

            # 6. Dạng "1. Nội dung"
            m = re.match(r"^(?!202\d)([1-9][0-9]{0,2})\.\s+(.+)", line)
            if m:
                if not inside_dieu:
                    result.append(f"#### {m.group(1)}.")
                    result.append(m.group(2))
                    last_number_heading_allowed = True
                else:
                    result.append(line)
                    last_number_heading_allowed = False
                i += 1
                continue

            # 7. Dạng "1.1 Nội dung"
            m = re.match(r"^([1-9]\d*(\.[1-9]\d*){1,2})\s+(.+)", line)
            if m:
                if last_number_heading_allowed:
                    result.append(f"##### {m.group(1)}")
                    result.append(m.group(3))
                else:
                    result.append(line)  # không markdown nếu số cha không được markdown
                i += 1
                continue

            # Nội dung thông thường
            result.append(line)
            i += 1

        processed_docs.append("\n".join(result))

    return processed_docs


def extract_legal_metadata(docs):

    all_metadata = []

    for doc in docs:
        lines = doc.strip().splitlines()
        lines = [line.strip() for line in lines if line.strip() != ""]

        metadata = {
            "co_quan_ban_hanh": "",
            "ma_so": "",
            "ngay_ban_hanh": "",
            "noi_ban_hanh": "",
            "loai_van_ban": "",
            "chu_de": ""
        }

        # 1. Cơ quan ban hành: dòng đầu tiên
        if len(lines) >= 1:
            metadata["co_quan_ban_hanh"] = lines[0].lower()

        # 2. Dòng thứ 3: chứa mã số và ngày ban hành
        if len(lines) >= 3:
            line3 = lines[2].strip()

            # Tách riêng mã số
            ma_so_match = re.search(r"Số\s*:\s*([\w\/\-]+)", line3, re.IGNORECASE)
            if ma_so_match:
                metadata["ma_so"] = ma_so_match.group(1).strip()

                # Loại bỏ phần "Số: xxx" để tránh dính vào nơi ban hành
                line3 = re.sub(r"Số\s*:\s*[\w\/\-]+", "", line3, flags=re.IGNORECASE).strip(" ,")

                # Loại bỏ từ mở đầu như "Luật", "Nghị định", v.v nếu có
                line3 = re.sub(r"^\b(luật|nghị[^\s]*)\b", "", line3, flags=re.IGNORECASE).strip()


            # Tách nơi ban hành và ngày
            date_match = re.search(
                r"(.*?),?\s*ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
                line3,
                re.IGNORECASE
            )
            if date_match:
                place = date_match.group(1).strip()
                day = int(date_match.group(2))
                month = int(date_match.group(3))
                year = int(date_match.group(4))

                metadata["noi_ban_hanh"] = place.title()
                metadata["ngay_ban_hanh"] = f"{day:02d}/{month:02d}/{year}"

        # 3. Loại văn bản: dòng thứ 4
        if len(lines) >= 4:
            loai = lines[3].strip()
            loai = re.sub(r"^#+\s*", "", loai)
            metadata["loai_van_ban"] = loai.title()

        # 4. Chủ đề: dòng thứ 5
        if len(lines) >= 5:
            metadata["chu_de"] = lines[4].strip().capitalize()

        all_metadata.append(metadata)

    return all_metadata


# raw_legal_docs = load_content_doc(jsonl_path)
# preprocessed_legal_docs = preprocess_text(raw_legal_docs)
# markdown_legal_docs = preprocess_legal_documents_to_markdown(preprocessed_legal_docs)
# legal_metadata = extract_legal_metadata(markdown_legal_docs)