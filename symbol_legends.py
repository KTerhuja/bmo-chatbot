import sys
import os
import pandas as pd
from collections import defaultdict
from pathlib import Path
import fitz
import numpy as np
from PIL import Image
import sqlite3
import re
import fasttext
import uuid 
import time

# Import custom utilities (Assuming these exist in your environment)
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR))
from utils_ifu.load_models import load_yolo_model
from utils_ifu.utils import (
    get_doc, get_subset_text_grid, normalize_text, scale_bbox_to_original, 
    cluster_arabic_sentences_multiline, select_sections_by_bbox, bbox_relative, 
    find_best_match, get_text_bbox, get_subset_others, filter_detections, get_subset_others_ar, np_array_to_base64
)
from utils_ifu.object_detection import run_object_detection
from utils_ifu.genai import parse_table_grid, parse_legends_booklet
from utils_ifu.extraction import (
    ocr_extract_misc_table, parse_legends_others, get_best_fit_bbox, 
    get_legends_others, get_legends_others_standard, get_legends_others_fallback
)
from utils_ifu.ocr_map import get_ocr_map
from utils.blob_storage import *
from object_detection.yolo_detect import *
from utils.ocr import run_document_intelligence
from utils_ifu.legend_preparation import filter_page_symbols
from utils_ifu.acclm_ifu import find_acclm_template_path


def ocr_object_detection(img, detections, client_ocr):
    ocr_map = get_ocr_map()
    ocr_corrected_list = []

    for det in detections:
        class_name = det.get("class_name")

        # Update the class_name if it exists in OCR map
        if class_name in ocr_map:
            x1, y1, x2, y2 = det["bbox"]
            img_array = np.array(img)
            cropped_img = img_array[y1:y2, x1:x2]
            
            # Convert to base64 for the OCR client
            base64_arr = np_array_to_base64(cropped_img)

            analyze_request = {"base64Source": base64_arr}

            poller = client_ocr.begin_analyze_document(
                "prebuilt-layout", body=analyze_request
            )

            result = poller.result()
            ocr_class = result.content.replace("\n", "_").replace(" ", "_")

            if ocr_class in ocr_map:
                det["class_name"] = ocr_map[ocr_class]
        if det not in ocr_corrected_list:
            ocr_corrected_list.append(det)

    # 3. FIX: Move print outside the loop so you only see the final result once
    print("Final Corrected List:", ocr_corrected_list)
    
    return ocr_corrected_list
def clean_page_nos(page_nos):
    """Removes outliers from page numbers using IQR."""
    if not page_nos: return []
    data = np.array(page_nos)
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)].tolist()

def get_page_image(doc, page_no, zoom=4):
    """Converts a PDF page to a PIL Image."""
    mat = fitz.Matrix(zoom, zoom)
    page = doc[page_no]
    pix = page.get_pixmap(matrix=mat)
    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    
    if pix.n > 1: # Convert to grayscale/RGB
        img_array = np.mean(img_array, axis=2).astype(np.uint8)
    
    return Image.fromarray(img_array).convert("RGB")

def last_two_of_run(lst):
    best_start, best_len = 0, 1
    start, length = 0, 1
    for i in range(1, len(lst)):
        if lst[i] == lst[i - 1] + 1:
            length += 1
            if length > best_len:
                best_start, best_len = start, length
        else:
            start, length = i, 1
    run = lst[best_start : best_start + best_len]
    return run[-2:]


def get_translation_map(cur, labels, lang):
    """Fetches translations in bulk for a specific language."""
    norm_labels = [normalize_text(l) for l in labels]
    placeholders = ",".join("?" for _ in norm_labels)
    query = f"SELECT label, translation FROM symbol_legends WHERE lang = ? AND label IN ({placeholders})"
    
    try:
        cur.execute(query, [lang] + norm_labels)
        return {row[0]: row[1] for row in cur.fetchall()}
    except sqlite3.OperationalError:
        # Fallback for massive labels sets
        mapping = {}
        for lbl in norm_labels:
            cur.execute("SELECT translation FROM symbol_legends WHERE label = ? AND lang = ?", (lbl, lang))
            row = cur.fetchone()
            if row: mapping[lbl] = row[0]
        return mapping
    

#code for pre-processing std-layout
def parse_legends_std(doc, page_nos):
    full_data = {}
    pattern = r'\b[a-z]{2}:[\s\S]*?(?=\b[a-z]{2}:|$)'
    for page_no in page_nos:
        page = doc[page_no]
        extracted_string = ""
        block_text = page.get_text("blocks")
        for block in block_text:
            text = block[4]
            extracted_string +=" "+text
        matches = re.findall(pattern, extracted_string)
        output = [re.sub(r'\s+', ' ', match).strip() for match in matches]
        full_data[page_no] = output

    return full_data

def parse_translations(input_dict):
    output = []
    current_bucket = None
    for page_no in sorted(input_dict.keys()):
        for item in input_dict[page_no]:

            parts = item.split(':', 1)
        
            if len(parts) != 2:
                continue 
                
            lang_code = parts[0].strip()
            text = parts[1].strip()

            if lang_code == 'en':
                if current_bucket is not None:
                    output.append(current_bucket)
            
                current_bucket = {
                    "legend": text,
                    "page_no": [page_no],
                    "translations": {}
                }
            else:
                # If it's not English, add it to the ongoing bucket
                if current_bucket is not None:
                    current_bucket["translations"][lang_code] = text
            
                    if page_no not in current_bucket["page_no"]:
                        current_bucket["page_no"].append(page_no)
    if current_bucket is not None:
        output.append(current_bucket)

    return output


def parse_translations(input_dict):
    output = []
    current_bucket = None
    
    for page_no in sorted(input_dict.keys()):
        for item in input_dict[page_no]:
            parts = item.split(':', 1)
       
            if len(parts) != 2:
                continue
                
            lang_code = parts[0].strip()
            text = parts[1].strip()

            if lang_code == 'en':
                if current_bucket is not None:
                    output.append(current_bucket)
           
                current_bucket = {
                    "legend": text,
                    "page_no": [page_no],
                    "translations": {} 
                }
            else:
                # If it's not English, add it to the ongoing bucket
                if current_bucket is not None:

                    if page_no not in current_bucket["translations"]:
                        current_bucket["translations"][page_no] = {}
                        
                    current_bucket["translations"][page_no][lang_code] = text
                    if page_no not in current_bucket["page_no"]:
                        current_bucket["page_no"].append(page_no)
                        
    if current_bucket is not None:
        output.append(current_bucket)

    return output
def transform_data(input_data):
    # Step 1: Group data by page_no
    pages_data = {}
    
    for item in input_data:
        legend = item.get("legend")
        # Grab the entire translations dictionary (keyed by language codes)
        translations = item.get("translations", {})
        
        for page in item.get("page_no", []):
            if page not in pages_data:
                pages_data[page] = []
                
            pages_data[page].append({
                "en_legend": legend,
                "translation": translations  # Directly assign the translations here
            })

    # Step 2: Build the final output structure with sequential indices
    output = {}
    
    for i, (page, metadata_list) in enumerate(pages_data.items()):
        # Create the metadata dictionary with sequential integer keys
        metadata_dict = {j: meta for j, meta in enumerate(metadata_list)}
        
        output[i] = {
            "page_no": page,
            "metadata": metadata_dict
        }
        
    return output


def get_img_path_by_page(image_paths, page_no):
    target_pattern = f"misc_{page_no}_symbols.png"
    
    for path in image_paths:
        if target_pattern in path:
            return path
            
    return None

def _process_grid_pdf(doc, data_json, yolo_model, cur, zoom, class_legend_mapping):
    """Handles logic for 'grid' type PDFs."""
    page_results, symbol_detection_results = {}, {}
    
    # 1. Get English Reference
    page_no_en, bbox_en = data_json['en']['metadata']['grids'][-1]
    extracted_data_en = get_subset_text_grid(doc, bbox_en, page_no_en, 'en')
    english_legend = [meta[4].strip() for meta in extracted_data_en]

    for lang, details in data_json.items():
        if lang in ('en', 'misc'): continue
        
        grid_idx = 0 if lang == 'ar' else -1
        page_no, bbox = details['metadata']['grids'][grid_idx]
        
        page_results[lang] = {"page_no": [page_no], "img_path": [details['metadata']['image_path']], "pass": [], "fail": []}
        symbol_detection_results[lang] = {"page_no": [page_no], "pass": [], "fail": []}

        # Fetch translations for this language
        trans_map = get_translation_map(cur, english_legend, lang)
        
        # Text Validation
        extracted_other = get_subset_text_grid(doc, bbox, page_no, lang)
        if len(extracted_other) == len(english_legend):
            for i, (x0, y0, x1, y1, text, _, _) in enumerate(extracted_other):
                label = normalize_text(english_legend[i])
                db_text = trans_map.get(label)
                
                text_bbox = [int(x * zoom) for x in bbox_relative([x0, y0, x1, y1], bbox)]
                res = {"bbox": text_bbox, "text": {"en": english_legend[i], lang: {"db_text": db_text, "extracted_text": text}}}
                
                if db_text and normalize_text(db_text) == normalize_text(text):
                    page_results[lang]["pass"].append(res)
                else:
                    page_results[lang]["fail"].append(res)

        # Object Detection
        img = get_page_image(doc, page_no, zoom)
        scaled_grid = [zoom * x for x in bbox]
        img = img.crop(scaled_grid)
        detections = run_object_detection(img, yolo_model)

        ## Object detection with OCR layer
        img_arr = np.array(img.convert('RGB'))
        img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        try:
            img_arr = cv2.GaussianBlur(img_bgr, (3, 3), 0)
            text_data = run_document_intelligence(img_arr)
        except:
            text_data = []
        
        detections = box_symbol_detection(img_bgr, text_data, detections)
        detections = detections_postprocessing(detections, img_bgr, text_data, acclm=True)
        detections = filter_page_symbols(doc, page_no, detections, grid=bbox, zoom=zoom)
        

        for det in detections:
            # legend = class_legend_mapping.get(det['class_name'], det['class_name'])
            legend = find_acclm_template_path(det['class_name'])
            entry = {"bbox": [int(x) for x in det['bbox']], "class": det['class_name'], "legend": legend, "validation": det['validations'][0] if det.get('validations') else None}
            target_list = "pass" if legend in english_legend else "fail"
            symbol_detection_results[lang][target_list].append(entry)

    return page_results, symbol_detection_results

def get_legends_symbols_others_standard(lan, data_json, doc, yolo_model):
    zoom = 4
    legends_full = []
    metadata = data_json.get(lan, {}).get("metadata", {})
    page_no = metadata.get("grids", [])
    page_no = clean_page_nos(page_no)
    last_page_no = page_no[-1]
    legends_full = get_legends_others_standard(doc, last_page_no)
    legends_metadata = {"legends_full":legends_full, "legends":[legends_full], "page_nos":[last_page_no]}
    ## object detection 
    mat = fitz.Matrix(zoom, zoom)

    page = doc[last_page_no]
    pix = page.get_pixmap(matrix=mat)

    img_data = pix.samples
    img_array = np.frombuffer(img_data, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    # Convert to grayscale if needed

    if pix.n > 1:

        img_array = np.mean(img_array, axis=2).astype(np.uint8)

    img = Image.fromarray(img_array).convert("RGB")

    print(f"Running object detection for page: {last_page_no}")

    detections_full = run_object_detection(img, yolo_model)

    detections_full = filter_detections(detections_full, legends_full)


    img_arr = np.array(img.convert('RGB'))
    try:
        img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        img_arr = cv2.GaussianBlur(img_bgr, (3, 3), 0)
        text_data = run_document_intelligence(img_arr)
    except:
        text_data = []
    
    detections_full = box_symbol_detection(img_array, text_data, detections_full)
    detections_full = detections_postprocessing(detections_full, img_array, text_data, acclm=True)
    detections_full = filter_page_symbols(doc, last_page_no, detections_full, zoom=zoom)

    symbols_metadata = {"symbols_full": detections_full, "symbols": [detections_full]}

 
    print(f"length of {lan}: {len(legends_full)}")

    return legends_metadata, symbols_metadata

def longest_subsequence(nums):
    if not nums:
        return []
    s = set(nums)
    best_start = None
    best_len = 0

    for x in s:
        # Start a run only at the beginning of a sequence
        if x - 1 not in s:
            y = x
            while y + 1 in s:
                y += 1
            length = y - x + 1
            if length > best_len or best_start is None:
                best_len = length
                best_start = x

    return list(range(best_start, best_start + best_len))


def get_language_legends_symbols_others(lan, data_json, doc, tables_metadata, yolo_model, client_ocr):
    """
    Handles legend and symbol extraction for 'others' PDF types, 
    managing multi-page logic and language-specific overrides.
    """
    zoom = 4
    metadata = data_json.get(lan, {}).get("metadata", {})
    page_nos = metadata.get("grids", [])
    if lan == "en":
        page_nos = last_two_of_run(page_nos)
    # 1. Handle Arabic Exception
    if lan == "ar":
        last_page_no = page_nos[-1]
        legends = get_subset_others_ar(doc, last_page_no)
        img = get_page_image(doc, last_page_no, 4)
        detections = filter_detections(run_object_detection(img, yolo_model), legends)
        # For Arabic in this branch, we skip the complex multi-page detection 
        # unless specifically required by the layout.
        return (
            {"legends_full": legends, "legends": [legends], "page_nos": [last_page_no]},
            {"symbols_full": detections, "symbols": [detections]}
        )

    # 2. Handle Multi-page or Single-page Standard Layouts
    image_paths = metadata.get("image_path", [])
    
    if len(image_paths) > 1 and len(page_nos) >= 2:
        # Process the last two pages (penultimate and last)
        target_pages = [page_nos[-2], page_nos[-1]]
        all_legends, all_detections = [], []
        page_wise_legends, page_wise_symbols = [], []

        for p_no in target_pages:
            print(f"Processing page {p_no} for language: {lan}")
            
            # Extract Text Legends
            page_legends = get_legends_others(p_no, doc, tables_metadata, lan)
            all_legends.extend(page_legends)
            page_wise_legends.append(page_legends)

            # Object Detection
            img = get_page_image(doc, p_no, zoom)
            detections = run_object_detection(img, yolo_model)
            time.sleep(10)
            ## Object detection with OCR layer
            mat = fitz.Matrix(zoom, zoom)
            page = doc[p_no]
            pix = page.get_pixmap(matrix=mat)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            
            if pix.n > 1: # Convert to grayscale/RGB
                img_array = np.mean(img_array, axis=2).astype(np.uint8)

            img_arr = np.array(img.convert('RGB'))
            try:
                img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
                img_arr = cv2.GaussianBlur(img_bgr, (3, 3), 0)
                text_data = run_document_intelligence(img_arr)
            except:
                text_data = []
            detections = box_symbol_detection(img_array, text_data, detections)
            detections = detections_postprocessing(detections, img_array, text_data, acclm=True)
            detections = filter_page_symbols(doc, p_no, detections, zoom=zoom)
            
            # filtered = filter_detections(detections, page_legends)
            # filtered = ocr_object_detection(img, filtered, client_ocr)
            # all_detections.extend(filtered)
            page_wise_symbols.append(detections)

        return (
            {"legends_full": all_legends, "legends": page_wise_legends, "page_nos": target_pages},
            {"full_symbols": all_detections, "symbols": page_wise_symbols}
        )

    elif page_nos:
        # Single-page fallback
        last_p = page_nos[-1]
        legends = get_legends_others(last_p, doc, tables_metadata, lan)
        
        img = get_page_image(doc, last_p, zoom)
        detections = filter_detections(run_object_detection(img, yolo_model), legends)
        time.sleep(10)
        ## Object detection with OCR layer
        mat = fitz.Matrix(zoom, zoom)
        page = doc[last_p]
        pix = page.get_pixmap(matrix=mat)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        if pix.n > 1: # Convert to grayscale/RGB
            img_array = np.mean(img_array, axis=2).astype(np.uint8)

        img_arr = np.array(img.convert('RGB'))
        try:
            img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
            img_arr = cv2.GaussianBlur(img_bgr, (3, 3), 0)
            text_data = run_document_intelligence(img_arr)
        except:
            text_data = []
        detections = box_symbol_detection(img_array, text_data, detections)
        detections = detections_postprocessing(detections, img_array, text_data, acclm=True)
        detections = filter_page_symbols(doc, last_p, detections, zoom=zoom)
        
        # detections = ocr_object_detection(img, detections, client_ocr)
        
        return (
            {"legends_full": legends, "legends": [legends], "page_nos": [last_p]},
            {"symbols_full": detections, "symbols": [detections]}
        )

    return {"legends_full": [], "legends": [], "page_nos": []}, {"symbols_full": [], "symbols": []}

def get_language_legends_symbols_others_fallback(data_json,lan, doc, page_no, pdf_path, mode,yolo_model, zoom=4):
    metadata = data_json.get(lan, {}).get("metadata", {})
    page_nos = metadata.get("grids", [])
    last_p = page_nos[-1]
    legends = get_legends_others_fallback(page_no, pdf_path, mode)
    
    img = get_page_image(doc, last_p, zoom)
    detections = filter_detections(run_object_detection(img, yolo_model), legends)
    ## Object detection with OCR layer
    mat = fitz.Matrix(zoom, zoom)
    page = doc[last_p]
    pix = page.get_pixmap(matrix=mat)
    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    
    if pix.n > 1: # Convert to grayscale/RGB
        img_array = np.mean(img_array, axis=2).astype(np.uint8)
    img_arr = np.array(img.convert('RGB'))
    try:
        img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        img_arr = cv2.GaussianBlur(img_bgr, (3, 3), 0)
        text_data = run_document_intelligence(img_arr)
    except:
        text_data = [] 
    detections = box_symbol_detection(img_array, text_data, detections)
    detections = detections_postprocessing(detections, img_array, text_data, acclm=True)
    detections = filter_page_symbols(doc, last_p, detections, zoom=zoom)
    
    
    return (
        {"legends_full": legends, "legends": [legends], "page_nos": [last_p]},
        {"symbols_full": detections, "symbols": [detections]}
        )


def deduplicate_detections(nested_data, descriptor_rows=None, zoom=4):
    """Filter by descriptor position; suppress only overlapping duplicate hits.

    Same-class symbols in separate legend rows are separate occurrences.
    descriptor_rows is page-aligned native PDF legend metadata.
    """
    from utils_ifu.symbol_geometry import filter_symbols_near_descriptors, suppress_overlapping_symbols
    if descriptor_rows is None:
        return [suppress_overlapping_symbols(items) for items in nested_data]
    if len(descriptor_rows) != len(nested_data):
        raise ValueError('Descriptor pages and detection pages must align')
    return [filter_symbols_near_descriptors(items, rows, zoom)
            for items,rows in zip(nested_data,descriptor_rows)]

from copy import deepcopy

def _bbox_pair_adjacent(b1, b2, y_tol=15.0, max_x_gap=80.0):
    """Check if two single bboxes sit on the same horizontal band and are
    side-by-side (no overlap, gap within max_x_gap)."""
    x0_1, y0_1, x1_1, y1_1 = b1
    x0_2, y0_2, x1_2, y1_2 = b2

    # Same vertical band: tops aligned AND bottoms aligned within tolerance
    if abs(y0_1 - y0_2) > y_tol:
        return False
    
    if abs(x1_1 - x0_2) < max_x_gap or abs(x1_2 - x0_1) < max_x_gap:
        return True
    return False


def _blocks_should_merge(block_a, block_b, y_tol=15.0, max_x_gap=80.0):
    """Two blocks merge if ANY bbox pair (one from each, on the same page)
    satisfies the adjacency rule."""
    for i, bbox_a in enumerate(block_a['bbox']):
        page_a = block_a['page_no'][i]
        for j, bbox_b in enumerate(block_b['bbox']):
            page_b = block_b['page_no'][j]
            if page_a != page_b:
                continue
            if _bbox_pair_adjacent(bbox_a, bbox_b, y_tol, max_x_gap):
                return True
    return False


def _merge_two(block_a, block_b):
    return {
        'bbox':    block_a['bbox']    + block_b['bbox'],
        'page_no': block_a['page_no'] + block_b['page_no'],
        'text':    block_a['text']    + block_b['text'],
    }


def merge_adjacent_blocks(blocks, y_tol=15.0, max_x_gap=80.0):
    """Iteratively merge any two blocks that have adjacent (side-by-side,
    same-row, same-page) bboxes. Repeats until no more merges are possible
    so that 3+ column groups also collapse correctly."""
    blocks = [deepcopy(b) for b in blocks]

    merged_in_pass = True
    while merged_in_pass:
        merged_in_pass = False
        i = 0
        while i < len(blocks):
            j = i + 1
            while j < len(blocks):
                if _blocks_should_merge(blocks[i], blocks[j], y_tol, max_x_gap):
                    blocks[i] = _merge_two(blocks[i], blocks[j])
                    blocks.pop(j)
                    merged_in_pass = True
                else:
                    j += 1
            i += 1
    return blocks

transformed_output = {}
page_counters = {}

def flush_block(page, en_text, translations):
    if page is None or en_text is None:
        return
    pg_str = page
    if pg_str not in transformed_output:
        transformed_output[pg_str] = {}
        page_counters[pg_str] = 1
    block_id = page_counters[pg_str]
    transformed_output[pg_str][block_id] = {
        'en': en_text,
        'translated': translations.copy()
    }
    page_counters[pg_str] += 1
def flush_transform_data(datamerged):    
    
    for item in datamerged:
        current_en = None
        current_page_translations = {}  # page -> {lang: text}
        
        for pg, parsed_dict in zip(item['page_no'], item['parsed']):
            if 'en' in parsed_dict and parsed_dict['en'] is not None:
                # Flush previous concept
                if current_en is not None:
                    for page, translations in current_page_translations.items():
                        flush_block(page, current_en, translations)
                    current_page_translations = {}
                
                current_en = parsed_dict['en']
                
                # Collect translations for THIS page only
                page_trans = {}
                for lang, text in parsed_dict.items():
                    if lang != 'en' and text is not None:
                        page_trans[lang] = text
                current_page_translations[pg] = page_trans
            else:
                # Append to the same page only
                page_trans = current_page_translations.get(pg, {})
                for lang, text in parsed_dict.items():
                    if lang != 'en' and text is not None:
                        page_trans[lang] = text
                current_page_translations[pg] = page_trans
                        
        if current_en is not None:
            for page, translations in current_page_translations.items():
                flush_block(page, current_en, translations)

    return transformed_output


def validate_symbols_and_detect_objects(
    client_ocr, llm_structured, llm, pdf_path: str, data_json: dict, db_path: str,
    yolo_model_path: str, class_legend_mapping: dict, zoom: int = 4, 
    pdf_type: str = None, mode: str = "blob"
):
    yolo_model = load_yolo_model(yolo_model_path)
    doc = get_doc(pdf_path, mode)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    lang_model_path = os.path.join(ROOT_DIR, 'ifu_models', 'lid.176.bin')
    lang_model = fasttext.load_model(lang_model_path)
    page_results, symbol_detection_results = {}, {}

    try:
        if pdf_type == "grid" and len(data_json['misc']['table']) < 2:
            page_results, symbol_detection_results = _process_grid_pdf(doc, data_json, yolo_model, cur, zoom, class_legend_mapping)
        
        elif pdf_type == "grid": # table logic
            # Extract data
            symbol_tables = data_json['misc']['table']
            meta_data = data_json['misc']['metadata']['grids']

            for i, symbol_table in enumerate(symbol_tables):
                page_no, grid = meta_data[i]
                page = doc[page_no]
                words_metadata = page.get_text("words")
                
                # Arabic fuzzy cluster prep
                arabic_cluster = cluster_arabic_sentences_multiline(words_metadata)
                full_arabic_text = [m[1] for m in arabic_cluster]
                words_metadata = select_sections_by_bbox(words_metadata, grid)

                symbol_legends_dict = parse_table_grid(symbol_table, llm_structured)
                page_results[i] = {"page_no": [page_no], "img_path": [symbol_table["image_path"]], "pass": defaultdict(list), "fail": defaultdict(list)}
                symbol_detection_results[i] = {"page_no": page_no, "pass": [], "fail": []}

                for j, en_legend in enumerate(symbol_legends_dict):
                    norm_en = normalize_text(en_legend)
                    for lang, legend in symbol_legends_dict[en_legend].items():
                        cur.execute("SELECT translation FROM symbol_legends WHERE label = ? AND lang = ?", (norm_en, lang))
                        row = cur.fetchone()
                        db_text = normalize_text(row[0]) if row else None
                        
                        # Bbox logic
                        search_text = f"{lang}-{legend}" if legend == "CONTROL RELEASE™" and lang != "ar" else legend
                        tb = get_text_bbox(words_metadata, search_text)
                        
                        if tb is None and lang == "ar":
                            _, _, idx = find_best_match(full_arabic_text, legend)
                            tb = arabic_cluster[idx][0]
                        
                        text_bbox = [int(x * zoom) for x in bbox_relative(list(tb), grid)] if tb else None
                        res = {"bbox": text_bbox, "text": {"en": en_legend, lang: {"db_text": db_text, "extracted_text": legend}}}
                        
                        target = "pass" if db_text == normalize_text(legend) else "fail"
                        page_results[i][target][j].append(res)

                # Object Detection
                img = get_page_image(doc, page_no, zoom)
                scaled_grid = [zoom * x for x in grid]
                img = img.crop(scaled_grid)
                detections = run_object_detection(img, yolo_model)
                ## Object detection with OCR layer
                img_arr = np.array(img.convert('RGB'))
                img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
                try:
                    img_arr = cv2.GaussianBlur(img_bgr, (3, 3), 0)
                    text_data = run_document_intelligence(img_arr)
                except:
                    text_data = [] 

                detections = box_symbol_detection(img_bgr, text_data, detections)
                detections = detections_postprocessing(detections, img_bgr, text_data, acclm=True)
                detections = filter_page_symbols(doc, page_no, detections, grid=grid, zoom=zoom)
                
                for det in detections:
                    # legend = class_legend_mapping.get(det['class_name'], det['class_name'])
                    legend = find_acclm_template_path(det['class_name'])
                    entry = {"bbox": [int(x) for x in det['bbox']], "class": det['class_name'], "legend": legend, "validation": det['validations'][0] if det.get('validations') else None}
                    target = "pass" if legend in symbol_legends_dict else "fail"
                    symbol_detection_results[i][target].append(entry)

        elif pdf_type == "others":
            width = doc[0].rect.width
            if len(data_json['misc']['table']) >= 3:
                if width > 700:
                    language_codes = list(data_json.keys())
                    language_codes.remove('misc')
                    page_nos = longest_subsequence(data_json['misc']['metadata']['grids'])[:-1]
                    print(f'page ranges from {page_nos}')

                    full_data = []
                    current_metadata = None

                    for page_no in page_nos:
                        page = doc[page_no]
                        block_text = page.get_text("blocks")
                        block_text = sorted(block_text, key=lambda item: (item[0] > 400, item[1], item[0]))

                        for block in block_text:
                            text = block[4]
                            x1, y1, x2, y2 = block[0:4]
                            bbox = [x1, y1, x2, y2]
                            lines = text.splitlines()

                            # Skip empty blocks to avoid IndexError on lines[0]
                            if not lines:
                                continue

                            lang_pred, prob_pred = lang_model.predict([lines[0]])
                            lan = lang_pred[0][0].replace('__label__', "")
                            prob = prob_pred[0][0]

                            # If language is English, start a new metadata dictionary
                            if lan == "en":
                                # If current_metadata already has data, append it to full_data before resetting
                                if current_metadata is not None:
                                    full_data.append(current_metadata)
                                
                                # Create a new metadata tracker
                                current_metadata = {"page_no": [], "bbox": [], "text": []}
                            elif current_metadata is None:
                                current_metadata = {"page_no": [], "bbox": [], "text": []}

                            # Append the current block's data to the active metadata dictionary
                            current_metadata["page_no"].append(page_no)
                            current_metadata["bbox"].append(bbox)
                            current_metadata["text"].append(text)

                    if current_metadata is not None:
                        full_data.append(current_metadata)

                    datamerged = merge_adjacent_blocks(full_data, y_tol=5, max_x_gap=40)
                    for metadata in datamerged:
                        output = parse_legends_booklet(llm,metadata['text'],language_codes)
                        print(output)
                        metadata["parsed"] = output

                    unique_id = str(uuid.uuid4())
                    english_legends = []
                    transformed_data = flush_transform_data(datamerged)
                    page_counter = list(transformed_data)
                    for i in range(len(page_counter)):
                        unique_id = str(uuid.uuid4())
                        page_no = page_counter[i]
                        mat = fitz.Matrix(zoom, zoom)
                        page = doc[page_no]
                        pix = page.get_pixmap(matrix=mat)
                        img_data = pix.samples
                        img_array = np.frombuffer(img_data, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                        # Convert to grayscale if needed
                        if pix.n > 1:
                            img_array = np.mean(img_array, axis=2).astype(np.uint8)

                        img = Image.fromarray(img_array).convert("RGB")
                        image_name = f"{unique_id}_{page_no}_symbols.png"

                        image_path = os.path.join("table_images", image_name)
                        
                        if mode == "local":
                            img.save(image_path, "PNG")
                        

                        elif mode == "blob":
                            img = np.array(img)
                            image_path = image_path.replace("\\","/")
                            print(image_path)
                            write_img(image_path, img)
                        

                        page_results[i] = {"page_no": [page_no], "img_path": [image_path], "pass": defaultdict(list), "fail": defaultdict(list)}

                        page = doc[page_no]
                        words_metadata = page.get_text("words")

                        for j, metadata in transformed_data[page_no].items():
                            norm_en = normalize_text(metadata['en'])
                            english_legends.append(norm_en)
                            for lang, legend in metadata['translated'].items():
                                norm_legend = normalize_text(legend)
                                cur.execute(
                                    "SELECT translation FROM symbol_legends WHERE label = ? AND lang = ?",
                                    (norm_en, lang),
                                )
                                row = cur.fetchone()
                                db_text = normalize_text(row[0]) if row else None
                                search_text = f"{lang}: {legend}" if legend == "CONTROL RELEASE™" and lang != "ar" else legend
                                tb = get_text_bbox(words_metadata, search_text)

                                text_bbox = [int(x * zoom) for x in tb] if tb else None
                                res = {
                                    "bbox": text_bbox,
                                    "text": {"en": norm_en, lang: {"db_text": db_text, "extracted_text": norm_legend}},
                                }
                                target = "pass" if db_text == legend else "fail"
                                page_results[i][target][j].append(res)

                    
                            
                            symbol_detection_results.setdefault(i, {"page_no": page_no, "pass": [], "fail": []})
                            print(f"Running object detection for grid: {i}")
                            detections = run_object_detection(img, yolo_model)
                            ## Object detection with OCR layer
                            mat = fitz.Matrix(zoom, zoom)
                            page = doc[page_no]
                            pix = page.get_pixmap(matrix=mat)
                            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                            
                            if pix.n > 1: # Convert to grayscale/RGB
                                img_array = np.mean(img_array, axis=2).astype(np.uint8)
                            img_arr = np.array(img.convert('RGB'))
                            try:
                                img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
                                img_arr = cv2.GaussianBlur(img_bgr, (3, 3), 0)
                                text_data = run_document_intelligence(img_arr)
                            except:
                                text_data = []
                            detections = box_symbol_detection(img_array, text_data, detections)
                            detections = detections_postprocessing(detections, img_array, text_data, acclm=True)
                            detections = filter_page_symbols(doc, page_no, detections, zoom=zoom)
                            
                            print("Object detection completed")

                            for detection in detections:
                                class_name = detection['class_name']
                                det_bbox = detection['bbox']
                                det_bbox = [int(x) for x in det_bbox]
                                validation = detection['validations'][0] if detection.get('validations') else None

                                # legend = class_legend_mapping.get(class_name, class_name)
                                legend = find_acclm_template_path(class_name)
                                if legend == class_name:
                                    print(f"Include {legend} to mapper")

                                if legend in english_legends:

                                    symbol_detection_results[i]["pass"].append({"bbox":det_bbox, "class": class_name, "legend":legend, "validation":validation})
                                else:
                                    symbol_detection_results[i]["fail"].append({"bbox":det_bbox, "class": class_name, "legend":legend, "validation":validation})

                else:         
                    page_nos = data_json['misc']['metadata']['grids']
                    full_data = parse_legends_std(doc, page_nos)
                    parsed_list = parse_translations(full_data)
                    output_dict = transform_data(parsed_list)
                    english_legends = []
                    image_paths = data_json['misc']['metadata']['image_path']
                    for metadata in parsed_list:
                        legend = metadata['legend']
                        english_legends.append(legend)
                        
                    for i, meta_data in output_dict.items():
                        page_no = meta_data["page_no"]
                        page = doc[page_no]
                        words_metadata = page.get_text("words")

                        img_path = get_img_path_by_page(image_paths, page_no)
                        page_results[i] = {"page_no": [page_no], "img_path": [img_path], "pass": defaultdict(list), "fail": defaultdict(list)}
                        # symbol_detection_results[i] = {"page_no": page_no, "pass": [], "fail": []}
                        for j, translations in output_dict[i]['metadata'].items():
                            norm_en = normalize_text(output_dict[i]['metadata'][j]['en_legend'])
                            for lang, legend in output_dict[i]['metadata'][j]['translation'][page_no].items():
                                norm_legend = normalize_text(legend)
                                cur.execute("SELECT translation FROM symbol_legends WHERE label = ? AND lang = ?", (norm_en, lang))
                                row = cur.fetchone()
                                db_text = normalize_text(row[0]) if row else None
                                search_text = f"{lang}: {legend}" if legend == "CONTROL RELEASE™" and lang != "ar" else legend
                                tb = get_text_bbox(words_metadata, search_text)

                                text_bbox = [int(x * zoom) for x in list(tb)] if tb else None
                                res = {"bbox": text_bbox, "text": {"en": norm_en, lang: {"db_text": db_text, "extracted_text": norm_legend}}}
                                
                                target = "pass" if db_text == legend  else "fail"
                                page_results[i][target][j].append(res)
                                
                        symbol_detection_results.setdefault(i, {"page_no": page_no, "pass": [], "fail": []})
                        mat = fitz.Matrix(zoom, zoom)
                        page = doc[page_no]
                        pix = page.get_pixmap(matrix=mat)
                        img_data = pix.samples
                        img_array = np.frombuffer(img_data, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

                        # Convert to grayscale if needed
                        if pix.n > 1:
                            img_array = np.mean(img_array, axis=2).astype(np.uint8)

                        img = Image.fromarray(img_array).convert("RGB")


                        print(f"Running object detection for grid: {i}")
                        detections = run_object_detection(img, yolo_model)
                        time.sleep(10)
                        ## Object detection with OCR layer
                        img_arr = np.array(img.convert('RGB'))
                        try:
                            img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
                            img_arr = cv2.GaussianBlur(img_bgr, (3, 3), 0)
                            text_data = run_document_intelligence(img_arr)
                        except:
                            text_data = []
                        detections = box_symbol_detection(img_array, text_data, detections)
                        detections = detections_postprocessing(detections, img_array, text_data, acclm =True)
                        detections = filter_page_symbols(doc, page_no, detections, zoom=zoom)
                        
                        print("Object detection completed")

                        for detection in detections:
                            class_name = detection['class_name']
                            det_bbox = detection['bbox']
                            validation = detection['validations'][0] if detection.get('validations') else None
                            det_bbox = [int(x) for x in det_bbox]

                            # legend = class_legend_mapping.get(class_name, class_name)
                            legend = find_acclm_template_path(class_name)
                            if legend == class_name:
                                print(f"Include {legend} to mapper")

                            if legend in english_legends:

                                symbol_detection_results[i]["pass"].append({"bbox":det_bbox, "class": class_name, "legend":legend, "validation":validation})
                            else:
                                symbol_detection_results[i]["fail"].append({"bbox":det_bbox, "class": class_name, "legend":legend, "validation":validation})

                
                                    
            else:
                en_meta = data_json.get('en', {}).get("metadata", {})
                last_en_page = clean_page_nos(en_meta.get("grids", []))[-1]

                # --- NEW: Check if standard legend path should be used ---
                legends_count = len(get_legends_others_standard(doc, last_en_page))
                use_standard = legends_count > 5


                # Get English Ref
                tables_metadata = {}

                for to_lan in data_json.keys():
                    if to_lan not in ["misc", "en"]:
                        for x in data_json[to_lan]["table"]:
                            try:
                                page_no = x['page_no'][0]
                            except:
                                page_no = x['page_no']
                            tables_metadata[page_no] = x["table_bbox"]

                if not use_standard:
                    tables_metadata = {}
                    tracker = {}
                    for lan in data_json.keys():
                        if lan != 'misc':
                            metadata = data_json.get(lan, {}).get("metadata", {})
                            page_nos = metadata.get("grids", [])
                            if lan == "en":
                                page_nos = last_two_of_run(page_nos)
                        
                            # 2. Handle Multi-page or Single-page Standard Layouts
                            image_paths = metadata.get("image_path", [])
                            
                            if len(image_paths) > 1 and len(page_nos) >= 2:
                                # Process the last two pages (penultimate and last)
                                target_pages = [page_nos[-2], page_nos[-1]]
                                all_legends = []

                                for p_no in target_pages:
                                    print(f"Processing page {p_no} for language: {lan}")
                                    
                                    # Extract Text Legends
                                    page_legends = get_legends_others(p_no, doc, tables_metadata, lan)
                                    all_legends.extend(page_legends)
                                    tracker[lan] = len(all_legends)

                            elif page_nos:
                                # Single-page fallback
                                last_p = page_nos[-1]
                                legends = get_legends_others(last_p, doc, tables_metadata, lan)
                                tracker[lan] = len(legends)

                    import statistics
                    # Check if "en" value is in the list of most frequent values
                    no_flag = tracker["en"] in statistics.multimode(tracker.values())

                # --- Choose extraction method based on standard check ---
                if use_standard:
                    en_l_meta,_ = get_legends_symbols_others_standard("en", data_json, doc, yolo_model)
                else:
                    if no_flag:
                        en_l_meta, _ = get_language_legends_symbols_others("en", data_json, doc, tables_metadata, yolo_model, client_ocr)
                    else:
                        en_l_meta,_ = get_language_legends_symbols_others_fallback(data_json,"en", doc, page_no, pdf_path, mode,yolo_model, zoom=4)

                en_labels = [m[4].strip() for m in en_l_meta['legends_full']]

                # --- NEW: Batch-fetch all translations upfront for efficiency ---
                needed_keys = set()
                for lan in data_json.keys():
                    if lan in ('en', 'misc'):
                        continue
                    for label in en_labels:
                        needed_keys.add((normalize_text(label), lan))

                translations = {}
                langs = sorted({lang for (_, lang) in needed_keys})

                for lang in langs:
                    labels_for_lang = [normalize_text(label) for (label, l) in needed_keys if l == lang]
                    placeholders = ",".join("?" for _ in labels_for_lang)
                    sql = f"SELECT label, translation FROM symbol_legends WHERE lang = ? AND label IN ({placeholders})"
                    params = [lang] + labels_for_lang
                    try:
                        cur.execute(sql, params)
                    except sqlite3.OperationalError:
                        # fallback: query per label
                        for label in labels_for_lang:
                            cur.execute(
                                "SELECT translation FROM symbol_legends WHERE label = ? AND lang = ?",
                                (label, lang)
                            )
                            row = cur.fetchone()
                            if row:
                                translations[(label, lang)] = row[0]
                        continue
                    for row in cur.fetchall():
                        lbl, trans = row
                        translations[(lbl, lang)] = trans

                # --- Process each language ---
                for lan in [l for l in data_json.keys() if l not in ('en', 'misc')]:

                    # --- Choose extraction method based on standard check ---
                    if use_standard:
                        l_meta, s_meta = get_legends_symbols_others_standard(lan, data_json, doc, yolo_model)
                    else:
                        l_meta, s_meta = get_language_legends_symbols_others(lan, data_json, doc, tables_metadata, yolo_model,client_ocr)

                    if len(l_meta["legends_full"]) != len(en_labels):
                        continue

                    # --- Use batch translations dict instead of per-language get_translation_map ---
                    image_paths = data_json[lan]['metadata'].get('image_path', [])
                    if lan != "ar":
                        image_paths = sorted(image_paths)
                    page_results[lan] = {
                        "page_no": l_meta['page_nos'],
                        "img_path": image_paths,
                        "pass": {},
                        "fail": {}
                    }
                    symbol_detection_results[lan] = {"pass": {}, "fail": {}}

                    global_idx = 0
                    for i, p_no in enumerate(l_meta['page_nos']):
                        p_pass, p_fail, s_pass, s_fail = [], [], [], []

                        for leg in l_meta["legends"][i]:
                            ref = normalize_text(en_labels[global_idx])
                            # --- Use batch translations lookup (supports both old and new path) ---
                            db_t = translations.get((ref, lan))
                            res = {
                                "bbox": [int(zoom * x) for x in leg[:4]],
                                "text": {
                                    "en": en_labels[global_idx],
                                    lan: {
                                        "db_text": db_t,
                                        "extracted_text": leg[4]
                                    }
                                }
                            }
                            if db_t and normalize_text(db_t) == normalize_text(leg[4]):
                                p_pass.append(res)
                            else:
                                p_fail.append(res)
                            global_idx += 1
                        detections = deduplicate_detections(s_meta['symbols'], l_meta['legends'], zoom)
                        for det in detections[i]:
                            # leg_name = class_legend_mapping.get(det['class_name'], det['class_name'])
                            leg_name = find_acclm_template_path(det['class_name'])
                            entry = {
                                "bbox": [int(x) for x in det['bbox']],
                                "class": det['class_name'],
                                "legend": leg_name,
                                "validation": det['validations'][0] if det.get('validations') else None
                            }
                            if leg_name in en_labels:
                                s_pass.append(entry)
                            else:
                                s_fail.append(entry)

                        page_results[lan]["pass"][p_no] = p_pass
                        page_results[lan]["fail"][p_no] = p_fail
                        symbol_detection_results[lan]["pass"][p_no] = s_pass
                        symbol_detection_results[lan]["fail"][p_no] = s_fail


    finally:
        doc.close()
        conn.close()

    return page_results, symbol_detection_results
