import arxiv
import fitz
import json
import re
import os
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFProcessor:
    @staticmethod
    def downloadPDF_by_id(id: str, path: str, name: str):
        search = arxiv.Search(id_list=[id])
        paper = next(search.results())
        paper.download_pdf(dirpath=path, filename=name+".pdf")

    @staticmethod
    def pdf_chunk(pdf_path, output_json_path):
        # 1. 路径检查
        output_dir = os.path.dirname(output_json_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if not os.path.exists(pdf_path):
            print(f"错误：找不到文件 {pdf_path}")
            return None

        # 2. 初始化转换器并转换
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        doc = result.document # 这是一个 DoclingDocument 对象
        
        elements_with_meta = []
        current_section = "Abstract/Introduction"

        # 3. 迭代元素（修复报错的关键）
        for element, _level in doc.iterate_items():
            # 获取页码
            page_num = 1
            if element.prov and len(element.prov) > 0:
                page_num = element.prov[0].page_no

            # --- 修改部分：不再使用 doc.export_to_markdown(obj=element) ---
            # 直接使用 element 对象转 markdown，或者获取其文本
            try:
                # 尝试使用元素自带的导出方法（如果有）
                content = doc.export_to_markdown(item=element) # 注意新版可能是 item
            except:
                # 兜底方案：如果是标题或正文，直接取其内容
                content = getattr(element, 'text', "")
            
            if not content or len(content.strip()) < 5:
                continue

            # 4. 识别标题以更新 Section
            # Docling 的标题元素通常有特定的标记，或者我们可以根据内容判断
            # 简单的判断逻辑：如果是标题层级
            from docling_core.types.doc.labels import DocItemLabel
            if element.label in [DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE]:
                current_section = content.strip()

            elements_with_meta.append({
                "content": content.strip(),
                "page": page_num,
                "section": current_section
            })

        # 5. 合并同页、同 Section 的内容进行分块
        # 这一步是为了防止 Chunk 太碎
        page_map = {}
        for item in elements_with_meta:
            key = (item["page"], item["section"])
            if key not in page_map:
                page_map[key] = []
            page_map[key].append(item["content"])

        # 6. 分块
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        final_json_data = []
        global_chunk_id = 0

        for (page, section), texts in page_map.items():
            combined_text = "\n\n".join(texts)
            sub_chunks = splitter.split_text(combined_text)
            
            for chunk_text in sub_chunks:
                final_json_data.append({
                    "chunk_id": global_chunk_id,
                    "content": chunk_text.strip(),
                    "metadata": {
                        "source": os.path.basename(pdf_path),
                        "page": page,
                        "section": section,
                    }
                })
                global_chunk_id += 1

        # 7. 保存
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_json_data, f, ensure_ascii=False, indent=4)
        
        print(f"解析成功！共生成 {global_chunk_id} 个块。")
        return final_json_data

# PDFProcessor.downloadPDF_by_id("2402.03300", "./data/pdfs/", "GRPO")
if __name__ == "__main__":
    result = PDFProcessor.pdf_chunk("./data/pdfs/GRPO.pdf", "./data/pdf_tmp/GRPO.json")