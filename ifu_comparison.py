from utils_ifu.comparison_core import (
    describe_diff,
    get_deviations,
    pair_strings_with_embeddings,
    sort_legends_by_layout,
    pair_legends,
    bucket_metadata,
    find_misspelled,
    add_bbox_columns,
    add_future_value_validation,
    restore_english_symbols
)
import difflib
import pandas as pd
import numpy as np
import time
from utils_ifu.embeddings import get_embeddings, cosine_sim
from utils_ifu.genai import parse_table_grid
from utils.blob_storage import read_file, write_img
from PIL import Image
import numpy as np
import cv2
import io
from utils_ifu.object_detection import run_object_detection
from utils_ifu.load_models import load_yolo_model
from utils_ifu.legends_map import get_legends_map
from utils_ifu.extraction import get_legends_others
from utils_ifu.symbol_legends import get_legends_others_standard
import os
from spellchecker import SpellChecker
import re
import os
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR))
from object_detection.yolo_detect import *
from utils.ocr import run_document_intelligence
from utils.utils import dict_get_none
from utils.utils import validations_join, validations_str, validations_append, value_accuracy_str
from utils_ifu.symbol_name_mapping import acclm_mapping
from utils_ifu.utils import clean_non_arabic_newlines, parse_legends
class_legend_mapping = acclm_mapping()
from utils_ifu.acclm_ifu import find_acclm_template_path




import sys
from collections import Counter
import fitz  # PyMuPDF
def remove_en_prefix(items):
    return [item.removeprefix("en-") for item in items]
def page_line_starts(page):
    """x0 of every text line on the page, as fraction of page width."""
    W = page.rect.width
    starts = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(s["text"] for s in line["spans"]).strip()
            if len(text) >= 3:                    # skip stray marks
                starts.append(line["bbox"][0] / W)
    return starts


def page_is_two_column(page):
    MID_LO, MID_HI = 0.40, 0.65   # where column 2 usually starts
    PAGE_THRESHOLD = 0.16   
    starts = page_line_starts(page)
    if len(starts) < 8:
        return None                               # not enough text to judge
    mid = sum(1 for s in starts if MID_LO <= s <= MID_HI)
    return mid / len(starts) >= PAGE_THRESHOLD


def to_key(x):
    return tuple(to_key(i) for i in x) if isinstance(x, (list, tuple)) else x

def dedupe(arr):
    seen = set()
    return [x for x in arr if not (to_key(x) in seen or seen.add(to_key(x)))]
def page_layout(doc, max_pages=40):
    # doc = fitz.open(path)
    idx = list(range(len(doc)))
    if len(idx) > max_pages:
        step = len(idx) / max_pages
        idx = [int(i * step) for i in range(max_pages)]

    votes = Counter(page_is_two_column(doc[p]) for p in idx)
    yes, no = votes[True], votes[False]
    if yes + no == 0:
        return "unknown", "no analyzable text (scanned PDF? try OCR first)"

    frac = yes / (yes + no)
    # detail = f"{yes}/{yes + no} sampled text pages have a two-column body"
    # Single-column docs produce ~0% two-column pages; booklets show a
    # substantial share even when figure/symbol/cover pages dilute it.
    if frac >= 0.5:
        return "booklet"
    if frac < 0.50:
        return "normal"




def drop_duplicates(nested):
    seen = set()
    unique = []
    for item in nested:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def extract_legends_others_ifu_comp(data_json, doc, tables_metadata, lan='en'):
    """
    Extract legends from the penultimate and last pages of a document.

    """
    page_nos = []
    legends_data_booklet = []
    legends_data = []
    legends_data_std = []
    # last_page_nos = data_json[lan]['metadata']['grids'][-3:]
    # if len(last_page_nos) < 3:
    #     last_page_nos = data_json[lan]['metadata']['grids']

    # start_page_nos = data_json[lan]['metadata']['grids'][:3]

    paths = data_json[lan]['metadata']['image_path']
    print(f"debug paths {paths}")
    last_page_nos = [int(re.search(r'(\d+)_symbols\.png$', p).group(1)) for p in paths]
    print(f"debug page nos {last_page_nos}")

    tables_metadata = {}
    legends_clean = []
    layout = page_layout(doc)
    if len(doc) <2:
        layout = "booklet"
    for page_no in last_page_nos:
        legends = get_legends_others(page_no, doc, tables_metadata, lan)
        if len(legends) > 2:
            page_nos.append(page_no)
            legends_data.append(legends)

        legends_standard = get_legends_others_standard(doc, page_no)
        if len(legends_standard) == 1 and "Federal law restricts this device" in legends_standard[0][4] and len(legends_data)<2:
            page_nos.append(page_no)
            legends_data_std.append(legends_standard)
        elif len(legends_standard) >= 3:
            page_nos.append(page_no)
            legends_data_std.append(legends_standard)

    if layout == "booklet":
        for legend_block in legends_data:
            for legend in legend_block:
                legends_clean.append(legend)
    elif layout == "normal":
        for legend_block in legends_data_std:
            for legend in legend_block:
                legends_clean.append(legend)
    temp = drop_duplicates(legends_clean)
    only_legends = [x[4] for x in temp]
    return only_legends




import numpy as np
import time


def get_translation_legends(data, lang):
    return [translations[lang] for translations in data.values() if lang in translations]

def get_legends_grid_comp(data_json, llm_structured, lang):
    symbol_tables = data_json['misc']['table']
    meta_data = data_json['misc']['metadata']['grids']    
    legends = []

    for i in range(len(symbol_tables)):
        symbol_legends_dict = parse_table_grid(symbol_tables[i], llm_structured)

        if lang == "en":
            for legend in list(symbol_legends_dict.keys()):
                legends.append(legend)
        else:
            translated_legends = get_translation_legends(symbol_legends_dict, lang)
            for legend in translated_legends:
                legends.append(legend)

    return legends

def split_path_parts(path):
    """Split a path on / or \\, ignoring empty segments."""
    return [p for p in re.split(r"[\\/]+", str(path)) if p]



def append_imgs(img_paths, mode,lan):
    unique_id = str(uuid.uuid4())

    if mode == "local":
        images = [Image.open(p) for p in img_paths]
    elif mode == "blob":
        images = [Image.open(io.BytesIO(read_file(p))) for p in img_paths]


    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    combined = Image.new('RGB', (total_width, max_height), (255, 255, 255))

    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    parts = split_path_parts(img_paths[0])

    file_name = parts[-2]        # the document folder, e.g. '391256E01_Redlined_11-13-2024 CURRENT'
    new_folder_path = parts[0]   # top-level dir, e.g. 'table_images'

    image_name = f"{file_name}_{unique_id}_{lan}_combined.png"

    image_path = os.path.join(new_folder_path, image_name)
    image_path = image_path.replace("\\","/")
    
    if mode == "local":

        combined.save(image_path)

    elif mode == "blob":

        write_img(image_path, np.array(combined))

    return image_path



def symbol_detect_comp(img_path, mode, yolo_model):
    symbol_detection_results= {}
    legends = []
    bbox = []
    acclm_validation = []
    if mode == "blob":
        img = Image.open(io.BytesIO(read_file(img_path)))
    elif mode == "local":
        img = Image.open(img_path)

    detections = run_object_detection(img, yolo_model)
    img_arr = np.array(img.convert('RGB'))
    # box_symbol_detection internally converts with COLOR_BGR2GRAY (and inspects text
    # brightness to decide when to invert for white-on-dark box symbols), so it must be
    # fed a BGR image exactly like process_pdf does. Passing RGB swaps the channels and
    # breaks that logic.
    img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    img_arr = cv2.GaussianBlur(img_bgr, (3, 3), 0)
    text_data = run_document_intelligence(img_arr)
    # Match process_pdf ordering: run box_symbol_detection FIRST so the box symbols it adds
    # are then post-processed (orientation + TEMP/N_PC/STACK_LIMIT/PIN_CONNECTOR/CE/RX
    # suffixing). Running detections_postprocessing first also consumes text_data entries
    # that box_symbol_detection relies on to match symbols.
    detections = box_symbol_detection(img_bgr, text_data, detections)
    detections = detections_postprocessing(detections, img_bgr, text_data, acclm = True)

    print("Object detection completed")

    for detection in detections:
        
        class_name = detection['class_name']
        det_bbox = detection['bbox']
        try:
            acclm_validation.append(detection["validations"][0])
        except:
            acclm_validation.append(None)
        det_bbox = [int(x) for x in det_bbox]
        legend = find_acclm_template_path(class_name)
        # legend = class_legend_mapping.get(class_name, class_name)        
        legends.append(legend)
        bbox.append(det_bbox)

        
    symbol_detection_results= {"legends":legends,
                                "future_value_validation" : acclm_validation,
                                "bbox":bbox,
                                "img_path": img_path}
    
    return symbol_detection_results


def build_comparison_df(current, future):
    """Align current/future legends row-by-row and flag deviations."""
    current_legends = current['legends']
    future_legends = future['legends']
    current_set = set(current_legends)

    n = max(len(current_legends), len(future_legends))
    rows = []
    for i in range(n):
        cur = current_legends[i] if i < len(current_legends) else None
        fut = future_legends[i] if i < len(future_legends) else None
        # A future legend deviates if it is not present in the current set.
        deviation = (fut is not None) and (fut not in current_set)
        rows.append({'current': cur, 'future': fut, 'deviation': deviation})

    return pd.DataFrame(rows)



import uuid  

def crop_datamatrix_img(img,bbox, mode, pdf_type):

    unique_id = str(uuid.uuid4())
    image_name = f"{unique_id}_datamatrix_{pdf_type}.png"
    folder_name = "datamatrix"
    os.makedirs(folder_name, exist_ok=True)

    image_path = os.path.join(folder_name, image_name)
    image_path = image_path.replace("\\","/")
    cropped_img = img.crop(bbox)
    
    if mode == "local":

        cropped_img.save(image_path)

    elif mode == "blob":

        write_img(image_path, np.array(cropped_img))

    return image_path
import fitz
from utils.barcode import detect_barcode_datamatrix
def make_comp_dict_datamatrix(from_barcode, to_barcode): 
    # Get previous validations (only TO barcode)
    validations_to = validations_join(to_barcode, None)
    validations = []
    
    barcode_type = dict_get_none(from_barcode, 'type')
    if not barcode_type:
        barcode_type = dict_get_none(to_barcode, 'type')

    expected_value = dict_get_none(to_barcode, 'expected_value')

    comp_dict = {
        'type': barcode_type,
        'from_bbox': dict_get_none(from_barcode, 'bbox'),
        'to_bbox': dict_get_none(to_barcode, 'bbox'),
        'from_decoded': dict_get_none(from_barcode, 'decoded'),
        'to_decoded': dict_get_none(to_barcode, 'decoded'),
        'deviation': char_diff(dict_get_none(from_barcode, 'decoded'),dict_get_none(to_barcode, 'decoded')),
        # 'validations': validations_str(validations) if from_barcode and to_barcode else None,
        'Future Barcode Value Accuracy':  value_accuracy_str(validations_to, expected_value=expected_value) if to_barcode else None,
        'validations_to': value_accuracy_str(validations_to, expected_value=expected_value) if to_barcode else None,
        'grade' : dict_get_none(to_barcode, 'grade'),
        'grade_metrics' : dict_get_none(to_barcode, 'grade_metrics'),
        'subtype' : dict_get_none(to_barcode, 'subtype')
    }
    return comp_dict
# def get_datamatrix_ifu(doc, pdf_type, mode, yolo_model):
#     if pdf_type == 'grid':
#         page_no = 0
#         dpi = 300 # DPI after zoom
#         zoom = dpi / 72
#         mat = fitz.Matrix(zoom, zoom)
#         pixels_to_cm = 2.54/dpi # cm per pixel
#         page = doc[page_no]
#         quad_max_size=1500

#     elif pdf_type == 'others':
#         zoom = 1
#         mat = fitz.Matrix(zoom, zoom)
#         pixels_to_cm = 2.54 / (72 * zoom)
#         page_no = len(doc) - 1
#         page = doc[page_no]
#         quad_max_size=2500

#     pix = page.get_pixmap(matrix=mat)
#     img_data = pix.samples
#     img_array = np.frombuffer(img_data, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
#     if pix.n > 1:
#         img_array = np.mean(img_array, axis=2).astype(np.uint8)

#     img = Image.fromarray(img_array).convert("RGB")
#     detections = run_object_detection(img, yolo_model, quad_max_size=quad_max_size)

#     # large image for display
#     zoom_display = 8
#     mat_display = fitz.Matrix(zoom_display, zoom_display)
#     page = doc[page_no]
#     pix = page.get_pixmap(matrix=mat_display)
#     img_data_display = pix.samples
#     img_array_display = np.frombuffer(img_data_display, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
#     if pix.n > 1:
#         img_array_display = np.mean(img_array_display, axis=2).astype(np.uint8)

#     img_display = Image.fromarray(img_array_display).convert("RGB")
#     scale_x = img_display.width  / img.width
#     scale_y = img_display.height / img.height


    
#     pixels_to_mm = pixels_to_cm * 10  # Convert cm/pixel to mm/pixel
#     for i, symbol in enumerate(detections):
#         if symbol['class_name'] in ['BARCODE', 'DATAMATRIX']:
#             bbox = symbol['bbox']
#             x1, y1, x2, y2 = bbox
#             bbox_display = [x1 * scale_x, y1 * scale_y,
#                             x2 * scale_x, y2 * scale_y]
#             img_path = crop_datamatrix_img(img_display,bbox_display, mode, pdf_type)
#             detected_codes = detect_barcode_datamatrix(img_array, bbox=bbox, btype=symbol['class_name'].lower(), pixels_to_mm=pixels_to_mm)
#             return detected_codes, img_path
        
#     return None, None
def _barcode_pixmap_gray(pix):
    """Keep the previous channel-average pixels without a full float64 image."""
    # samples_mv borrows the pixmap buffer instead of copying its RGB bytes.
    # The returned grayscale array owns its memory after the pixmap is released.
    samples = np.frombuffer(pix.samples_mv, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 1:
        return samples[:, :, 0].copy()
    gray = np.empty((pix.height, pix.width), dtype=np.uint8)
    rows_per_chunk = max(1, (1024 * 1024) // max(1, pix.width))
    for start in range(0, pix.height, rows_per_chunk):
        stop = min(start + rows_per_chunk, pix.height)
        channel_sum = np.add.reduce(samples[start:stop], axis=2, dtype=np.uint16)
        np.floor_divide(channel_sum, pix.n, out=channel_sum)
        gray[start:stop] = channel_sum
        del channel_sum
    return gray


def _render_datamatrix_crop(page, bbox, detection_size):
    """Render just the preview crop at the existing 8x resolution."""
    zoom = 8
    matrix = fitz.Matrix(zoom, zoom)
    display_bounds = (page.rect * matrix).irect
    scale_x = display_bounds.width / detection_size[0]
    scale_y = display_bounds.height / detection_size[1]
    # Match PIL's rounding of the old full-page image crop. Include a small
    # border when rendering so the final crop keeps the same pixel alignment.
    left, top, right, bottom = [int(round(value)) for value in (
        bbox[0] * scale_x, bbox[1] * scale_y,
        bbox[2] * scale_x, bbox[3] * scale_y)]
    clip = fitz.Rect(left + display_bounds.x0 - 1, top + display_bounds.y0 - 1,
                     right + display_bounds.x0 + 1, bottom + display_bounds.y0 + 1) / zoom
    pix = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
    gray = _barcode_pixmap_gray(pix)
    offset_x, offset_y = display_bounds.x0 - pix.x, display_bounds.y0 - pix.y
    with Image.fromarray(gray) as rendered:
        with rendered.crop((left + offset_x, top + offset_y,
                            right + offset_x, bottom + offset_y)) as cropped:
            return cropped.convert("RGB")


def get_datamatrix_ifu(doc, pdf_type, mode, yolo_model):

    def render_page_gray(page, zoom):
        """Render a page at the given zoom and return a grayscale numpy array."""
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return _barcode_pixmap_gray(pix)

    if pdf_type == 'grid':
        page_no = 0
        dpi = 300  # DPI after zoom
        zoom = dpi / 72
        pixels_to_cm = 2.54 / dpi  # cm per pixel
        page = doc[page_no]
        quad_max_size = 1500

        img_array = render_page_gray(page, zoom)
        with Image.fromarray(img_array).convert("RGB") as img:
            detections = run_object_detection(img, yolo_model, quad_max_size=quad_max_size)

    elif pdf_type == 'others':
        page_no = len(doc) - 1
        page = doc[page_no]
        quad_max_size = 2500

        detections = None
        img_array = None
        pixels_to_cm = None

        # Try zoom values 1 to 4 until a DATAMATRIX is detected
        for zoom in range(1, 5):
            candidate_array = render_page_gray(page, zoom)
            with Image.fromarray(candidate_array).convert("RGB") as candidate_img:
                candidate_detections = run_object_detection(candidate_img, yolo_model, quad_max_size=quad_max_size)

            if any(d['class_name'] == 'DATAMATRIX' for d in candidate_detections):
                detections = candidate_detections
                img_array = candidate_array
                pixels_to_cm = 2.54 / (72 * zoom)
                break
            # Release the rejected raster before allocating the next zoom.
            del candidate_array

        # No DATAMATRIX found at any zoom level -> exit
        if detections is None:
            return None, None

    else:
        return None, None

    pixels_to_mm = pixels_to_cm * 10  # Convert cm/pixel to mm/pixel
    for i, symbol in enumerate(detections):
        if symbol['class_name'] in ['DATAMATRIX']:
            bbox = symbol['bbox']
            with _render_datamatrix_crop(page, bbox, (img_array.shape[1], img_array.shape[0])) as preview:
                img_path = crop_datamatrix_img(preview, (0, 0, *preview.size), mode, pdf_type)
            detected_codes = detect_barcode_datamatrix(
                img_array, bbox=bbox,
                btype=symbol['class_name'].lower(),
                pixels_to_mm=pixels_to_mm
            )
            return detected_codes, img_path

    return None, None
def char_diff(word1: str, word2: str) -> str:
    try:
        if word1 is None and word2 is None:
            return "None."
        if word1 is None:
            return f"Current is None, Future is '{word2}'."
        if word2 is None:
            return f"Future is None, Current is '{word1}'."

        matcher = difflib.SequenceMatcher(None, word1, word2)
        changes = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "replace":
                changes.append(f"'{word1[i1:i2]}' changed to '{word2[j1:j2]}'")
            elif tag == "delete":
                changes.append(f"'{word1[i1:i2]}' was removed")
            elif tag == "insert":
                changes.append(f"'{word2[j1:j2]}' was inserted")

        if not changes:
            return "No differences found."

        return ", ".join(changes)
    except:
        return None





def get_grid_imgs(doc, legends_grid, mode):
    images = []
    zoom = 4
    for page_no, bbox in legends_grid:
        mat = fitz.Matrix(zoom, zoom)
        page = doc[page_no]
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.samples
        img_array = np.frombuffer(img_data, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        # Convert to grayscale if needed
        if pix.n > 1:
            img_array = np.mean(img_array, axis=2).astype(np.uint8)

        img = Image.fromarray(img_array).convert("RGB")

        scaled_grid = [zoom * x for x in bbox]
        cropped_img = img.crop(scaled_grid)
        images.append(cropped_img)

    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    combined = Image.new('RGB', (total_width, max_height), (255, 255, 255))

    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width

    unique_id = str(uuid.uuid4())
    new_folder_path = "table_images"
    image_name = f"{unique_id}_combined.png"

    image_path = os.path.join(new_folder_path, image_name)
    image_path = image_path.replace("\\","/")

    if mode == "local":

        combined.save(image_path)

    elif mode == "blob":

        write_img(image_path, np.array(combined))

    return image_path

import re
from typing import List, Tuple

# (x0, y0, x1, y1, text)
Block = Tuple[float, float, float, float, str]


def _sentence_count(text: str) -> int:
    """Number of real sentences in `text`.

    Splits on . ! ? that are followed by whitespace or end-of-string, so
    decimals ("0.7-8.0") and list markers ("1. ") do not inflate the count
    (single-token fragments are ignored).
    """
    parts = re.split(r'[.!?](?:\s|$)', text.strip())
    real = sum(1 for p in parts if len(p.split()) >= 2)
    return real or (1 if text.strip() else 0)


def extract_features(blocks: List[Block]) -> dict:
    texts = blocks
    n = len(texts)
    if n == 0:
        return {}

    word_counts = [len(t.split()) for t in texts]

    avg_words = sum(word_counts) / n
    # entries that read like a full sentence (end in . ! or ?)
    period_ratio = sum(1 for t in texts if t[-1] in '.!?') / n
    # entries containing more than one sentence
    multi_sentence_ratio = sum(1 for t in texts if _sentence_count(t) > 1) / n
    # paragraph-length entries
    long_ratio = sum(1 for wc in word_counts if wc > 15) / n

    return {
        "n_entries": n,
        "avg_words": round(avg_words, 2),
        "period_ratio": round(period_ratio, 2),
        "multi_sentence_ratio": round(multi_sentence_ratio, 2),
        "long_ratio": round(long_ratio, 2),
    }


def is_symbol_glossary(blocks: List[Block]) -> bool:
    """True  -> short symbol/label glossary   (keep, like examples 1 & 3)
       False -> body prose / paragraphs        (false positive, like example 2)"""
    f = extract_features(blocks)
    if not f:
        return False

    # Each test is TRUE for a glossary; prose breaks several at once.
    votes = (
        (f["avg_words"] < 9)              # short entries on average
        + (f["period_ratio"] < 0.25)      # few full sentences
        + (f["multi_sentence_ratio"] < 0.15)  # rarely >1 sentence per entry
        + (f["long_ratio"] < 0.20)        # few paragraph-length entries
    )
    return votes >= 3  





import re


def parse_translations(text: str) -> list[dict[str, str]]:
    records = []

    for part in text.split("/"):
        part = part.strip()
        if not part:
            continue

        match = re.fullmatch(
            r"(zh-cn|zh-tw|[a-z]{2})-(.*)",
            part,
            flags=re.DOTALL,
        )

        if match:
            language, translation = match.groups()
        elif part.endswith("-ar"):
            language, translation = "ar", part[:-3]
        else:
            continue  # Skip malformed entries.

        if language == "ar":
            records.append({})

        if not records:
            raise ValueError("Expected an Arabic entry first.")

        records[-1][language] = translation.strip()

    return records


# input_text is your original string.

def parse_grid_ifu_comp(df):
    ISO_639_1 = [
    "aa","ab","ae","af","ak","am","an","ar","as","av","ay","az","ba","be",
    "bg","bh","bi","bm","bn","bo","br","bs","ca","ce","ch","co","cr","cs",
    "cu","cv","cy","da","de","dv","dz","ee","el","en","eo","es","et","eu",
    "fa","ff","fi","fj","fo","fr","fy","ga","gd","gl","gn","gu","gv","ha",
    "he","hi","ho","hr","ht","hu","hy","hz","ia","id","ie","ig","ii","ik",
    "io","is","it","iu","ja","jv","ka","kg","ki","kj","kk","kl","km","kn",
    "ko","kr","ks","ku","kv","kw","ky","la","lb","lg","li","ln","lo","lt",
    "lu","lv","mg","mh","mi","mk","ml","mn","mr","ms","mt","my","na","nb",
    "nd","ne","ng","nl","nn","no","nr","nv","ny","oc","oj","om","or","os",
    "pa","pi","pl","ps","pt","qu","rm","rn","ro","ru","rw","sa","sc","sd",
    "se","sg","si","sk","sl","sm","sn","so","sq","sr","ss","st","su","sv",
    "sw","ta","te","tg","th","ti","tk","tl","tn","to","tr","ts","tt","tw",
    "ty","ug","uk","ur","uz","ve","vi","vo","wa","wo","xh","yi","yo","za",
    "zh","zu","zh-ch","zh-tk"
    ]
    LANGS = "(?:" + "|".join(ISO_639_1) + ")"
    target = "CONTROL RELEASE™"

    df = df["table"]
    df = df.dropna(axis=1)
    df.loc[-1] = df.columns
    df = df.iloc[:,:2]
    df.index = df.index + 1
    df = df.sort_index().reset_index(drop=True)
    df.columns = ['symbol', 'description']

    values = df['description'].values

    text = values[0]

    # 3. Iterate through the remaining items
    for item in values[1:]:
        # Check if the current item starts with 2 characters followed by a hyphen
        # r'^\w{2}-' matches "01-", "AA-", "1b-", etc.
        if re.match(r'^\w{2}-', item):
            separator = ' / '
        else:
            separator = ' '
        
        # Append the separator and the item
        text += separator + item
    text = clean_non_arabic_newlines(text)

    pattern = rf"(?<!/)\s+(?={LANGS}(?:-[a-z]{{2}})?-)"
    text = re.sub(pattern, " / ", text)
    pattern = r'(zh-tw-[^\u0600-\u06FF]*?)\s+(?=[\u0600-\u06FF])'
    text = re.sub(pattern, r"\1 / ", text)
    pattern = r'(?<=[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF])ar(?=[\s/])'
    text = re.sub(pattern, '-ar', text)

    text = text.replace("/ /", "/")
    print(text)
    result = parse_translations(text)

    return result

def get_translation_legends(data, lang):
    return [value[lang] for value in data if lang in value]
def get_legends_grid_comp(data_json, lang):

    if lang == "zh-hans":
        lang = "zh-ch"
    elif lang == "zh-hant":
        lang = "zh-tw"
    symbol_tables = data_json['misc']['table']
    meta_data = data_json['misc']['metadata']['grids']    
    legends = []

    for i in range(len(symbol_tables)):
        symbol_legends_dict = parse_grid_ifu_comp(symbol_tables[i])
        translated_legends = get_translation_legends(symbol_legends_dict, lang)
        legends.extend(translated_legends)

    return legends

from collections import defaultdict, deque



from math import isclose

def common_bboxes(symbols_legends_grid, grids, tolerance=1e-6):
    grids_by_page = {}
    for page_no, bbox in grids:
        grids_by_page.setdefault(page_no, []).append(bbox)

    return [
        [page_no, bbox]
        for page_no, bboxes in symbols_legends_grid.items()
        for bbox in bboxes
        if any(
            len(bbox) == len(other)
            and all(
                isclose(a, b, rel_tol=0, abs_tol=tolerance)
                for a, b in zip(bbox, other)
            )
            for other in grids_by_page.get(page_no, [])
        )
    ]
