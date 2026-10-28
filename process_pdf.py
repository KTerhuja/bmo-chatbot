import os
import sys
import re
import json
import unicodedata
from time import time
from io import BytesIO

import fitz
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import distance
from rapidfuzz import fuzz

ROOT_DIR = Path(__file__).parent.absolute() # path to repo
sys.path.append(str(ROOT_DIR)) 
from utils.utils import rect_to_list, show_img, remove_blue_lines, dict_set_none, dict_get_none
from utils.utils import validations_join, validations_str, validations_append, value_accuracy_str
from utils.utils import PageError, InputFileError, make_json_serializable, Timer, split_pdf, font_check, font_check_block
from utils.barcode import detect_barcode_datamatrix, save_barcode_crop#, grade_barcode
from utils.symbol import decode_symbol, get_symbol_business_name, SYMBOL_FONTS, BOX_SYMBOLS
from utils.data import load_vantage_file, load_translation_file, load_tad_file
from utils.data import load_ps5_file, load_manufacturer_address_db, TAD_COLS, MASTER_DATA_VER
from utils.color import get_common_colors, pantone_to_rgb, compare_colors, int_to_rgba, same_color, get_text_color
from utils.spatial import match_points, bbox_center, percent_distance, circular_distance, draw_dist_pct, bbox_distance, auto_align_pages, align_center, align_bbox
from utils.spatial import orientation_neighbors, join_rects, expand_bboxes, merge_elements, find_neighbor
from utils.spatial import is_next_line, find_next_line, find_breaks_line, bbox_inside_bbox, bbox_inside_bbox_soft
from utils.spatial import find_element_inside_bbox
from utils.blob_storage import write_img, read_pdf, read_file, read_docx
from utils.ocr import run_document_intelligence, run_document_intelligence_pdf 
from object_detection.yolo_detect import load_yolo_model, detect_objects, detections_postprocessing, box_symbol_detection, dark_box_symbol_detection, circle_symbol_detection, strand_symbol_detection, acclm_validate_symbol, logo_detection


# Symbols that must meet a minimum printed size. The regulatory conformity marks
# (CE, UKCA and the Ukraine "UA Conformity Sign") have to be at least
# MIN_SYMBOL_SIZE_MM in both width and height. A detected symbol whose name starts
# with any of these prefixes and falls below that size gets a 'symbol_size_error'
# flagged in its validations. Edit this list / value to change the rule.
MIN_SYMBOL_SIZE_MM = 5  # Minimum width and height (mm) for size-constrained symbols
SIZE_CONSTRAINED_SYMBOL_PREFIXES = [
    'CE',    # CE mark and numbered variants (e.g. 'CE', 'CE_2797_H')
    'UKCA',  # UKCA mark and numbered variants
    'UA',    # Ukraine "UA Conformity Sign"
]


# Latin typographic ligatures that some PDFs encode as a single glyph. When a
# word such as "fi" is stored as the ligature "ﬁ" (U+FB01), plain text matching
# downstream (language identifiers, symbols, TAD values, ...) fails. Normalising
# these back to their component letters at the source keeps everything working.
_LIGATURE_MAP = {
    '\ufb00': 'ff',   # ﬀ
    '\ufb01': 'fi',   # ﬁ
    '\ufb02': 'fl',   # ﬂ
    '\ufb03': 'ffi',  # ﬃ
    '\ufb04': 'ffl',  # ﬄ
    '\ufb05': 'st',   # ﬅ (long s + t)
    '\ufb06': 'st',   # ﬆ
    '\u0132': 'IJ',   # Ĳ
    '\u0133': 'ij',   # ĳ
    '\u0152': 'OE',   # Œ
    '\u0153': 'oe',   # œ
    '\u00c6': 'AE',   # Æ
    '\u00e6': 'ae',   # æ
}

_LIGATURE_RE = re.compile('|'.join(map(re.escape, _LIGATURE_MAP)))


def normalize_ligatures(text):
    """
    Replace Latin typographic ligatures (e.g. ﬁ, ﬂ, ﬀ) with their component
    letters so downstream text matching is not thrown off by single-glyph
    ligatures embedded in the PDF.
    """
    if not text:
        return text
    return _LIGATURE_RE.sub(lambda m: _LIGATURE_MAP[m.group(0)], text)


class ProcessPDF():
    def __init__(self, from_path, to_path, tad_path, output_folder, document_type, business_unit, mode="local"):
        self.timer = Timer()

        # Placeholder for docx redline and annotations
        self.redline_doc = None
        self.docx_annotations = None
        self.business_unit = business_unit # SURGERY, EP, VISION

        # Execution without TAD file if not provided
        if not tad_path:
            self.tadless = True
        else:
            self.tadless = False

        self.mode = mode
        if mode=='local':
            if to_path:

                # From
                if Path(from_path).suffix.lower() == ".docx":
                    self.from_doc, self.redline_doc, self.docx_annotations = read_docx(from_path, local = True)
                elif Path(from_path).suffix.lower() == ".pdf":
                    self.from_doc = fitz.open(from_path)

                # To
                if Path(to_path).suffix.lower() == ".docx":
                    self.to_doc, _, _ = read_docx(to_path, local = True) # Future should not have revisions
                elif Path(to_path).suffix.lower() == ".pdf":
                    self.to_doc = fitz.open(to_path)

            else: # No to_path provided, split the pdf on the fly
                self.from_doc, self.to_doc = split_pdf(from_path)
            if not self.tadless:
                # Load TAD file  
                self.df_tad = load_tad_file(tad_path) 
            
        elif mode=='blob':
            # From
            if Path(from_path).suffix.lower() == ".docx":
                self.from_doc, self.redline_doc, self.docx_annotations = read_docx(from_path)
            elif Path(from_path).suffix.lower() == ".pdf":
                self.from_doc = read_pdf(from_path)

            # To
            if Path(to_path).suffix.lower() == ".docx":
                self.to_doc, _, _ = read_docx(to_path) # Future should not have revisions
            elif Path(to_path).suffix.lower() == ".pdf":
                self.to_doc = read_pdf(to_path)

            # TAD
            if not self.tadless:
                self.df_tad = load_tad_file(BytesIO(read_file(tad_path)))

        self.colname_map = {v: k for k, v in TAD_COLS.items()} # Invert key:value pairs

        # Load stationary data
        self.load_data()

        # Clean "Sample" watermarks
        for page in self.to_doc:
            for xref in page.get_contents():
                stream = self.to_doc.xref_stream(xref).replace(b'Sample', b'')
                self.to_doc.update_stream(xref, stream)
        for page in self.from_doc:
            for xref in page.get_contents():
                stream = self.from_doc.xref_stream(xref).replace(b'Sample', b'')
                self.from_doc.update_stream(xref, stream)

        # How many pages to process, for debugging
        self.pages = False
        self.startpage = 1

        # Specific pages to process
        self.page_lists = (None, None) # (to_page_list, from_page_list), 1-based indexing

        # Method to use for detecting needle point symbols
        self.np_symbol_method = 'YOLO' # TM for template matching, or YOLO. TM is no longer supported
        self.yolo_weights = "best.pt" # Weights to use if np_symbol_method is YOLO

        # Use OCR on each page
        self.ocr = True

        self.document_type = document_type # Pre-Printed, Thunderbird, IFU comparison or IFU translation

        # Preprinted document. Used to trigger the extraction of RMC
        self.preprinted = document_type=="Pre-Printed" 
        self.ifu_comp = document_type=="IFU comparison" 

        # Store the brand name. Used to determine if code should keep looking for the brand
        self.brand_name = None

        # Used to store the extracted wave color from the vantage file. 
        # Used only when a page has no match in the vantage file, then this one is used. 
        self.wave_color_vantage = None

        # Bounds for the upper right label when there is a multilabel on the page. False means no multilabel
        self.upper_right_bbox = False

        # Store all detected fonts for debugging
        self.detected_fonts = set([])

        # Create output folder
        if not os.path.exists(output_folder) and self.mode=='local':
            os.makedirs(output_folder)
        self.output_folder = output_folder


        # A dict to return as a response for the API
        self.response = {'pages':[], 'success':True, 
                         "master_data_versions" : MASTER_DATA_VER,
                         "output_folder" : output_folder}

        self.timer.append("ProcessPDF init")

    def load_data(self):
        if self.business_unit == "SURGERY":
            # Load Vantage data
            vantage_path = "data/Thunderbird_DB.parquet"
            wave_colour_path = "data/Wave Color Source File.xlsx"
            self.df_vantage,  self.vantage_unique = load_vantage_file(os.path.join(ROOT_DIR,  vantage_path), os.path.join(ROOT_DIR, wave_colour_path))

            # Load PS-5 data
            ps5_path = "data/PS-0000005v22 Process Specification for Creation and Revision for Needles_Final.xlsx"
            self.df_ps5 = load_ps5_file(os.path.join(ROOT_DIR, ps5_path))

            # Load translation data
            trans_path = "data/FGA Master Translations List_04.03.25.xlsx"
            self.translation_database = load_translation_file(os.path.join(ROOT_DIR, trans_path))

            # Load manufacturer address db
            manu_path = "data/manufacturer_database.csv"
            self.manufacturer_address_db = load_manufacturer_address_db(os.path.join(ROOT_DIR, manu_path))

    def load_yolo(self):
        """
        Load yolo model for symbol detection
        """
        model_path = os.path.join(ROOT_DIR, f"object_detection/{self.yolo_weights}") 
        self.yolo_model = load_yolo_model(model_path)

    def process_all_pages(self):
        # Iterate over pages and compare nth page with nth page
        start = self.startpage -1
        end = min(start + self.pages, len(self.from_doc)) if self.pages else len(self.from_doc)
        
        if self.np_symbol_method == 'YOLO':
            self.load_yolo()

        # If page_lists is provided, validate and override iteration
        to_page_list, from_page_list = self.page_lists
        if to_page_list and from_page_list:
            to_page_list, from_page_list = self.page_lists
            if len(to_page_list) != len(from_page_list):
                raise InputFileError('page_lists: both lists must be of the same length.')
            to_page_list = sorted([p-1 for p in to_page_list])  # Convert to 0-based index and sort
            from_page_list = sorted([p-1 for p in from_page_list])  # Convert to 0-based index and sort
            if any(p>=len(self.to_doc) or p<0 for p in to_page_list) or any(p>=len(self.from_doc) or p<0 for p in from_page_list):
                raise InputFileError('page_lists: page numbers must be within the range of the documents.')
            page_pairs = list(zip(to_page_list, from_page_list))
        else:
            # Check that input files have the same number of pages
            if len(self.from_doc) != len(self.to_doc):
                raise InputFileError('The number of pages does not match between CURRENT and FUTURE documents.')
            page_pairs = [(i, i) for i in range(start, end)]

        from utils.process_pdf_ep import ProcessPage_EP, ComparePages_EP # Import here to avoid circular import. ProcessPage_EP needs ProcessPDF, and ProcessPDF needs ProcessPage_EP
        from utils.process_pdf_vision import ProcessPage_VISION, ComparePages_VISION # Import here to avoid circular import. ProcessPage_VISION needs ProcessPDF, and ProcessPDF needs ProcessPage_VISION
        
        BU_classes = {
            "SURGERY": (ProcessPage, ComparePages),
            "VISION": (ProcessPage_VISION, ComparePages_VISION),
            "EP": (ProcessPage_EP, ComparePages_EP),
        }

        for to_page_num, from_page_num in page_pairs:
            try:
                t0 = time()
                print("\nPage", to_page_num+1)
                print("TO:")
                to_page = self.to_doc.load_page(to_page_num)
                process_page_to = BU_classes[self.business_unit][0](to_page, page_type="to")
                process_page_to.process_pdf = self
                self.timer.append("stuff")  
                process_page_to.process()      

                print("FROM:")
                from_page = self.from_doc.load_page(from_page_num)
                process_page_from = BU_classes[self.business_unit][0](from_page, page_type="from")
                process_page_from.process_pdf = self
                # Pass the matching TAD and PS5 rows from To to From
                # if not self.tadless:
                if self.business_unit == "SURGERY":
                    process_page_from.tad_data = process_page_to.tad_data
                    process_page_from.ps5_data = process_page_to.ps5_data
                    process_page_from.vantage_data = process_page_to.vantage_data
                self.timer.append("stuff")                       
                process_page_from.process()

                print("COMPARE:")
                compare = BU_classes[self.business_unit][1](process_page_from, process_page_to)
                compare.process_pdf = self
                compare.compare()

                process_page_to.save_images(self.output_folder, to_page_num+1)
                process_page_from.save_images(self.output_folder, to_page_num+1)
                self.timer.append("save_images")                       

                #compare.show_comparison(output_folder)
                compare.response["processing_time"] = time() - t0
                self.response["pages"].append(compare.response)

                # Join all detected fonts 
                self.detected_fonts.update(process_page_to.detected_fonts)
                self.detected_fonts.update(process_page_from.detected_fonts)

                # Preprinted information
                if self.preprinted and from_page_num == 0 and self.business_unit=="SURGERY": # First page typically contains the preprinted info, and SURGERY is the only BU with preprinted for now
                    self.response['preprinted_info'] = {'RMC' : process_page_to.RMC,
                                                        'LAB' : process_page_to.LAB}
                    
                # Brand name
                self.response["brand_name"] = self.brand_name
                
            except PageError as e:
                try: 
                    to_ccn = self.to_page.ccn
                except AttributeError:
                    to_ccn = None
                response = {"success" : False,
                            "error_message" : e.message,
                            "CCN" : {'to':to_ccn}}
                response = make_json_serializable(response)
                self.response["pages"].append(response)
        self.timer.end()
        if self.mode=='local':
            self.save_response_json()
    
    def save_response_json(self):
        response_path = os.path.join(self.output_folder, "response.json")
        with open(response_path, 'w') as f:
            json.dump(self.response, f, default=str, indent=4)

class ProcessPage():
    def __init__(self, page, page_type):
  
        self.page = page
        self.page_type = page_type
        self.images = {}
        self.page_num = page.number

        self.sub_images = []

        self.needle_data = []
        self.barcodes = []
        self.symbol_data = []  

        self.ps5_data = None
        self.vantage_data = None
        self.tad_data = None

        # Create matrix for upscaling page images
        dpi = 300 # DPI after zoom
        zoom = dpi / 72
        self.mat = fitz.Matrix(zoom, zoom)
        self.pixels_to_cm = 2.54/dpi # cm per pixel

        # Page orientation
        # All extracted rects must be transformed by this
        # The pixmap (image) does not need this transform, that's why both mat and rot_mat are kept
        m = page.rotation_matrix
        # Correct the translation after rotation
        self.rot_mat = m * zoom

    def process(self):
        if self.page_type=="from":
            if not self.process_pdf.docx_annotations:
                self.extract_annotations()
            else:
                if self.process_pdf.redline_doc:
                    redline_page = self.process_pdf.redline_doc.load_page(self.page_num)
                    orig_pix = redline_page.get_pixmap(matrix=self.mat)
                    image = np.frombuffer(orig_pix.samples, dtype=np.uint8).reshape(orig_pix.height, orig_pix.width, orig_pix.n)
                    self.images["annotations"] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                self.annotations = [a for a in self.process_pdf.docx_annotations if a["page"]==self.page_num+1]
        else:
            self.annotations = []
        self.process_pdf.timer.append("extract_annotations")
        self.read_page(self.page)
        self.process_pdf.timer.append("read_page")  
        self.detect_grid()
        self.process_pdf.timer.append("detect_grid")  
        self.get_eurocode()
        if self.page_type=="to" and self.ccn:
            self.get_code_data()
        self.process_pdf.timer.append("get_eurocode")  
        if self.process_pdf.preprinted and self.page_type=="to":
            self.extract_RMC()
        if self.process_pdf.ocr:
            self.run_ocr()
            self.process_pdf.timer.append("run_ocr")
        if self.process_pdf.np_symbol_method == 'YOLO':
            # Symbols that never appear in SURGERY labels and should be ignored
            ignore_symbols = []
            # SURGERY is the only BU that validates symbols against the ACCLM library
            self.symbols_yolo(ignore_symbols=ignore_symbols, acclm=True)
            self.process_pdf.timer.append("symbols_yolo") 
        if self.process_pdf.ocr:
            self.cleanup_ocr()
            self.process_pdf.timer.append("cleanup_ocr")
        self.find_input_elements()
        self.process_pdf.timer.append("find_input_elements")  
        if self.page_type=="to" and self.ccn:
            self.validate_databases()
        self.find_translations()
        self.process_pdf.timer.append("find_translations")  
        self.find_addresses()
        self.process_pdf.timer.append("find_addresses")  
        self.decode_barcodes()
        self.process_pdf.timer.append("decode_barcodes")  
        self.validate_barcodes()
        self.process_pdf.timer.append("validate_barcodes")  
        if not self.process_pdf.brand_name: # Determine brand name if not done so already. Tipically first page will handle that.
            self.find_brand_name()
        self.find_wave_color()
        self.process_pdf.timer.append("wave_color")  
        pass # For debugging breakpoints

    def read_page(self, page):
        """
        Read page contents and store them in their respective attributes.
        Create page images.
        Separate pdf elements into layers:
        -Original layer
        -No text layer
        """

        def line_dir_orientation(line_dir, font=None, bbox=None):
            # Convert direction vector to orientation angle
            if line_dir == (1.0, 0.0):
                orientation = 0
            elif line_dir == (0.0, 1.0):
                orientation = 90
            elif line_dir == (-1.0, 0.0):
                orientation = 180
            elif line_dir == (0.0, -1.0):
                orientation = 270
            else:
                orientation = 0

            # Fallback for Type3 fonts with unreliable dir (e.g., 'Unnamed-T3').
            # Use bbox aspect ratio to infer the true orientation.
            if font and 'T3' in font and bbox is not None:
                w = abs(bbox[2] - bbox[0])
                h = abs(bbox[3] - bbox[1])
                if w > 0 and h > 0:
                    if h / w > 3 and orientation in [0, 180]:
                        orientation = 90
                    elif w / h > 3 and orientation in [90, 270]:
                        orientation = 0
            return orientation
            
        orig_pix = page.get_pixmap(matrix=self.mat)
        image = np.frombuffer(orig_pix.samples, dtype=np.uint8).reshape(orig_pix.height, orig_pix.width, orig_pix.n)
        self.images["original"] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.h, self.w = image.shape[:2]
        self.images["annotated"] = self.images["original"].copy()
        
        # Extract text, bounding boxes, and fonts
        self.text_data = []
        self.detected_fonts = []
        for block in page.get_text("dict")["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    line_dir = line['dir']
                    for span in line["spans"]:
                        if span["text"].strip(): # Check for empty strings
                            font = span["font"]
                            self.detected_fonts.append(font)
                            bbox = rect_to_list(fitz.Rect(span["bbox"]) * self.rot_mat)
                            orientation = line_dir_orientation(line_dir, font, bbox)
                            # Barcode in text font
                            if "UNBRNormal" in span["font"] or "UNCWNormal" in span["font"] or "UNBWNormal" in span["font"]:
                                # barcode will be detected by yolo and decoded at a later step
                                continue # This prevents deletion of this text

                            # Datamatrix in text font
                            elif "AZTWNormal" in span["font"]:
                                # datamatrix will be detected by yolo and decoded at a later step
                                continue # This prevents deletion of this text

                            # Normal Text
                            elif font.split('-')[0] not in SYMBOL_FONTS and 'ethicon' not in font.lower():
                                self.text_data.append({
                                    "text": normalize_ligatures(span["text"].replace(u'\xa0', u' ').replace(u'\x03', u' ')), # Fix non-breaking and EOT characters that are present in some translations and should be a space, and split any single-glyph ligatures (e.g. ﬁ -> fi)
                                    "bbox": bbox,
                                    "center": bbox_center(bbox),
                                    "font": font,
                                    "size" : span["size"],
                                    "color" : int_to_rgba(span["color"]),
                                    "orientation": orientation
                                })
                            
                            # Delete the text from the page
                            rect = fitz.Rect(span["bbox"])
                            # Add a redaction box that is a line (changed to a very narrow box) that goes in the direction of the text, to minimize overlap 
                            # with other elements (barcode and datamatix text in particular)
                            if orientation in [0, 180]:
                                rect = fitz.Rect(rect.x0, (rect.y0+rect.y1)/2-2, rect.x1, (rect.y0+rect.y1)/2+2)
                            else:
                                rect = fitz.Rect((rect.x0+rect.x1)/2-2, rect.y0, (rect.x0+rect.x1)/2+2, rect.y1)
                            page.add_redact_annot(rect)

        # Extract symbols. Needs another loop to extract individual character bounding boxes
        for block in page.get_text("rawdict")["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    line_dir = line['dir']
                    orientation = line_dir_orientation(line_dir)
                    for span in line["spans"]:
                        for char in span.get("chars", []):
                            font = span["font"].split('-')[0] # Keep only font family (e.g. discard "-Bold")
                            
                            if (font in SYMBOL_FONTS or 'ethicon' in font.lower()) and char not in ["", " "]:
                                bbox = fitz.Rect(char["bbox"])
                                # The following block is to correct bounding boxes of some rotated characters that are misrepresented in pymupdf
                                if line['dir'] == (-1.0, 0.0) and (char["origin"][1] - char["bbox"][1]) > (char["bbox"][3] - char["origin"][1]):
                                    # It was discovered that in such boxes the 'origin' Y coordinate is closer to ymax (should be closer yo ymin)
                                    bbox = [bbox[0],
                                            char["origin"][1],
                                            bbox[2],
                                            char["origin"][1]+bbox[3]-bbox[1]
                                            ]
                                    bbox = fitz.Rect(bbox)
                                    
                                bbox = rect_to_list(bbox * self.rot_mat)
                                
                                char_text = char["c"]  # The character itself
                                symbol_name, symbol_type = decode_symbol(char_text, font)
                                
                                if not symbol_name: # Do not store non-printable characters
                                    continue
                                if symbol_type=="Literal": # Treat logos as text
                                    self.text_data.append({
                                        "text": symbol_name,
                                        "bbox": bbox,
                                        "center" : bbox_center(bbox),
                                        "font": font,
                                        "size" : span["size"],
                                        "color" : int_to_rgba(span["color"]),
                                        "orientation": orientation
                                        })      
                                         
                                else:
                                    self.symbol_data.append({
                                        "symbol_name": symbol_name,
                                        "symbol_type": symbol_type,
                                        "bbox": bbox,
                                        "center" : bbox_center(bbox),
                                        "size" : span["size"],
                                        "font": font,
                                        "color" : int_to_rgba(span["color"]),
                                        "orientation": orientation
                                        })  
                                if symbol_type=="Unknown" and self.process_pdf.mode=='local':
                                    symbol_crop = self.images["original"][bbox[1]:bbox[3], bbox[0]:bbox[2]]   
                                    write_path = os.path.join('unknown_symbols', symbol_name+'.png')
                                    if not os.path.exists(write_path):
                                        cv2.imwrite(os.path.join('unknown_symbols', symbol_name+'.png'), symbol_crop)        

        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

        # Keep only unique fonts
        self.detected_fonts = set(self.detected_fonts)

        # Store the text layer    
        notext_pix = page.get_pixmap(matrix=self.mat)
        image = np.frombuffer(notext_pix.samples, dtype=np.uint8).reshape(notext_pix.height, notext_pix.width, notext_pix.n)
        self.images["no_text"] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return True
    
    def extract_annotations(self):
        """
        Extracts all the annotations (comments, cross-outs, etc.) present in the page.
        Some elements have more than one annotation, for example one Inserted Tesxt annotation 
        is often related to a cross-out. These are grouped for clarity, using irt_xref.
        Since the annotations come in a generator object in fitz, we have to iterate over it once to read, 
        and a second time to filter out.
        """
        self.annotations = []
        annotations = [] # Intermediate list
        saved_image = False
        # Extract and remove annotations
        for annot in self.page.annots():
            if not saved_image: # Save image with the annotations before removing them if there is at least one 
                orig_pix = self.page.get_pixmap(matrix=self.mat)
                image = np.frombuffer(orig_pix.samples, dtype=np.uint8).reshape(orig_pix.height, orig_pix.width, orig_pix.n)
                self.images["annotations"] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                saved_image = True

            annotations.append({
                "content": annot.info["content"],
                "title": annot.info["title"],
                "type": annot.info["subject"],
                "validation": "N/A",
                # "type": annot.type[1],
                "bbox": rect_to_list(annot.rect * self.rot_mat),
                "xref": annot.xref, # xref of this annotation
                "irt_xref": annot.irt_xref, # xref of the parent
            })
            self.page.delete_annot(annot)
        
        # Handle the annotation hierarchy
        # irt_xref indicates that an annotation is related to another one 
        for annot in annotations:
            # Ignore stamps that are inside another annotation (these are not linked with irt_xref, for some reason)
            if annot['type'] == "Stamp": 
                # Check if stamp is inside any other annotation
                inside_of = [ a for a in annotations if bbox_inside_bbox(annot['bbox'], a['bbox']) and annot!=a]
                if inside_of:
                    continue
            if annot['irt_xref'] == 0: # Has no parent
                children = [ a for a in annotations if a['irt_xref']==annot['xref']]
                for c in children:
                    annot['bbox'] = expand_bboxes(annot['bbox'], c['bbox']) # Join the two boxes
                    annot['content'] = annot['content'] + c['content'] # Join the content
                self.annotations.append(annot)

    def detect_grid(self):
        breaks_x = find_breaks_line(self.text_data, 0, self.w)
        breaks_y = find_breaks_line(self.text_data, 1, self.h)
        n_labels = (len(breaks_x)+1) * (len(breaks_y)+1)
        if n_labels >= 2:
            print('Multilabel detected')
            # Get upper right label bounds
            x1 = breaks_x[-1] if breaks_x else 0 # for when breaks_x = [] (vertical layout)
            y2 = breaks_y[0] if breaks_y else self.h # for when breaks_y = [] (horizontal layout)
            
            self.upper_right_bbox = [x1, 0, self.w, y2]

            # Delete all data outside the bounds
            # Text
            new_text_data = []
            for e in self.text_data:
                if bbox_inside_bbox(e['bbox'], self.upper_right_bbox):
                    new_text_data.append(e)
            self.text_data = new_text_data
            # Symbols
            new_symbol_data = []
            for e in self.symbol_data:
                if bbox_inside_bbox(e['bbox'], self.upper_right_bbox):
                    new_symbol_data.append(e)
            self.symbol_data = new_symbol_data
            # Barcodes
            new_barcodes = []
            for e in self.barcodes:
                if bbox_inside_bbox(e['bbox'], self.upper_right_bbox):
                    new_barcodes.append(e)
            self.barcodes = new_barcodes            
        else:
            self.upper_right_bbox = None

    def extract_RMC(self):
        # RMC
        suffixes = ['', ' :', ' #:']
        text_matches, match_idxs = self.search_in_text('RMC', suffixes=suffixes)
        if text_matches:
            el, match_idx = find_next_line(text_matches[0], self.text_data, thresh=15, mode='right')
            self.RMC = el['text'].strip() if el else 'Not found'
        else:
            self.RMC = 'Not Found'
        # LAB # EX. LAB12345678v2 -> starts with LAB, digits, a 'v', then digits
        expression = r"LAB\d+v\d+"
        text_matches, match_idxs, captured_vals = self.search_in_text_re(expression, case=True, partial_match=True)
        if text_matches:
            match = re.search(expression, text_matches[0]['text'])
            self.LAB = match.group(0) if match else 'Not Found'
        else:
            self.LAB = 'Not Found'

    def run_ocr(self):
        if self.upper_right_bbox: # Multilabel, only run upper right label
            urbb = self.upper_right_bbox
            ocr_image = self.images["no_text"][urbb[1]:urbb[3], urbb[0]:urbb[2], :]
        else:
            ocr_image = self.images["no_text"]
        ocr_text_data = run_document_intelligence(ocr_image)
        if self.upper_right_bbox: # Correct bboxes to original coordinates
            for det in ocr_text_data:
                det['bbox'] = [det['bbox'][0] + urbb[0],
                              det['bbox'][1] + urbb[1],
                              det['bbox'][2] + urbb[0],
                              det['bbox'][3] + urbb[1]]
        self.text_data.extend(ocr_text_data)

    def run_ocr_new(self):
        single_page_doc = fitz.open()   # new empty PDF
        if self.page_type == "to":
            single_page_doc.insert_pdf(self.process_pdf.to_doc, from_page=self.page_num, to_page=0)
        else:
            single_page_doc.insert_pdf(self.process_pdf.from_doc, from_page=self.page_num, to_page=0)
        page_bytes = single_page_doc.tobytes()
        ocr_text_data = run_document_intelligence_pdf(page_bytes, self.pixels_to_cm, self.images["no_text"])
        self.text_data.extend(ocr_text_data)
        pass

    def cleanup_ocr(self):
        """
        Delete any OCR text that is inside symbols and vice-versa (only relevant for YOLO-detected symbols)

        Symbols inside OCR are typically false positives
        OCR inside symbols is sometimes useful for fine-tuning symbol 
        classification (already used in symbols_yolo() and thus disposable/redundant)

        The number inside boxes like N_PC/STACK_LIMIT is already consumed and embedded in
        the symbol name during symbols_yolo() (n_pc_post/stack_limit_post), so no special
        casing is needed here anymore.

        """
        remove_ocr = []
        remove_sym = []
        for i, text in enumerate(self.text_data):
            if text['font'] == 'OCR_text':
                for j, symbol in enumerate(self.symbol_data):
                    # OCR text inside symbol?
                    if bbox_inside_bbox_soft(text['bbox'], symbol['bbox']): 
                        remove_ocr.append(i)
                    # Symbol inside OCR?
                    if bbox_inside_bbox_soft(symbol['bbox'], text['bbox']):
                        remove_sym.append(j)

        remove_ocr = list(set(remove_ocr))
        for i in sorted(remove_ocr)[::-1]:
            self.text_data.pop(i)

        remove_sym = list(set(remove_sym))
        for j in sorted(remove_sym)[::-1]: 
            self.symbol_data.pop(j)

    def find_input_elements(self):
        """
        Iterate over TAD columns and try search for matches in the label.
        It uses a different treatment for each column

        Tadless: Use a mix of regular expressions and database data to look for label elements

        For symbols and tadless, we can skip, since symbols are detected and identified already.
        """
        tad_matches = {}
        tad_results = {}
        all_match_idxs = []
        sep = ' /&/ ' # Separator for multiple values in TAD file

        # Brand # EX. ETHIBOND -> Text OR Literal Symbol
        element = 'brand'
        if not self.tad_data:
            match_idxs = []
            for unique_value in self.process_pdf.vantage_unique[element]:
                suffixes = ["TM", ""]
                _ , brand_match_idxs = self.search_in_text(unique_value, suffixes=suffixes, partial_match=True)
                match_idxs += brand_match_idxs
            # The above process could return duplicate matches because some brands share a root (e.g. "PDS" and "PDS PLUS")
            # To prevent duplicate entries from being added we delete duplicates and retrieve the matches directly from self.text_data
            match_idxs = list(set(match_idxs))
            tad_matches[element] = [self.text_data[idx] for idx in match_idxs]
        else:
            tad_value = self.tad_data[element]
            suffixes = ["TM", ""]
            tad_matches[element], match_idxs = self.search_in_text(tad_value, suffixes=suffixes, partial_match=True)
            tad_results[element] = bool(tad_matches[element])
            tad_matches[element] = [{**m, 'validations':['tad_pass']} for m in tad_matches[element]]
        all_match_idxs.extend(match_idxs)
        

        # Control Release # EX. "N-->No" -> Symbol
        # Add logic to match the CR found symbol to "To Number of Sutures" (e.g. CR_8 -> 8)
        element = 'ctrl_release'
        if self.tad_data:
            tad_value = self.tad_data[element] != "N-->No"
            search = 'CR' #all symbols that start with CR
            tad_matches[element] = self.search_in_symbols(search)
            tad_results[element] = bool(tad_matches[element]) == tad_value
            tad_matches[element] = [{**m, 'validations':['tad_data_error']} if not tad_value else {**m, 'validations':['tad_pass']} for m in tad_matches[element]] # Determine if tad_data_error is present

        # CE mark # EX. CE -> Symbol
        element = 'CE_to'
        if self.tad_data:
            tad_value = self.tad_data[element] in ["CE"]
            search = ['CE']
            tad_matches[element] = self.search_in_symbols(search)
            tad_results[element] = bool(tad_matches[element]) == tad_value
            tad_matches[element] = [{**m, 'validations':['tad_data_error']} if not tad_value else {**m, 'validations':['tad_pass']} for m in tad_matches[element]] # Determine if tad_data_error is present
        
        # Suture Color # EX. "WHITE /&/ GREEN" -> Symbol
        element = 'sut_color_to'
        if self.tad_data:
            tad_value = self.tad_data[element].split(sep) # separate values like "WHITE /&/ GREEN" in a tuple
            search = ["SA_UNDYED" if val.strip()=="WHITE" else "SA_DYED"  for val in tad_value]
            tad_matches[element] = [match for s in search for match in self.search_in_symbols(s)]
            tad_results[element] = all( any(match["symbol_name"].startswith(s) for match in tad_matches[element]) for s in search )

        #Number of strands # EX. 4 -> "4 x 30" (70cm)"
        # The number of strands is not labeled on its own; it is inferred from the context
        # "<n> x <length>" (e.g. "4 x 30\" (70cm)"). Capture the leading number that is
        # followed by an 'x' and a suture length (a number optionally followed by " or cm/in).
        element = 'num_strands_to'
        expression = r'(?<!\d)(\d{1,2})\s*x\s*\d+\s*(?:"|in|cm)'
        tad_matches[element], match_idxs, captured_vals = self.search_in_text_re(expression, case=False)
        if self.tad_data:
            tad_value = str(self.tad_data[element]).strip()
            tad_results[element] = all([c == tad_value for c in captured_vals])
            tad_matches[element] = [{**m, 'validations':['tad_data_error']} if tad_value!=m['captured_value'] else {**m, 'validations':['tad_pass']} for m in tad_matches[element]] # Determine if tad_data_error is present
        all_match_idxs.extend(match_idxs)

        # Length CM # EX. 75CM -> 75cm
        element = 'len_cm_to'
        # A 1 to 3 digit number followed by cm or CM (or cms)
        expression = r"(?<!\d)(\d{1,3})\s?cm"
        tad_matches[element], match_idxs, captured_vals = self.search_in_text_re(expression, case=False)
        if self.tad_data:
            tad_value = self.tad_data[element].lower().replace('cm', '') # Remove suffix for cases where it is missing (if any)
            if not tad_matches[element]: #Try to find the length isolated, for when 90 and cm are different boxes
                suffixes = [""]
                tad_matches[element], match_idxs = self.search_in_text(tad_value, suffixes=suffixes)
                tad_results[element] = bool(tad_matches[element])
            else:
                tad_matches[element] = [{**m, 'validations':['tad_data_error']} if tad_value!=m['captured_value'] else {**m, 'validations':['tad_pass']} for m in tad_matches[element]] # Determine if tad_data_error is present
                tad_results[element] = all([c==tad_value for c in captured_vals])
        all_match_idxs.extend(match_idxs)
        
        # Length Inches # EX. 30IN -> 30"
        element = 'len_in_to'
        # Similar to previous one, but followed by in or "
        expression = r'(?<!\d)(\d{1,3})\s*(in|\")'
        tad_matches[element], match_idxs, captured_vals = self.search_in_text_re(expression, case=False)
        if self.tad_data:
            tad_value = self.tad_data[element].lower().replace('in', '') # Remove suffix for cases where it is missing (if any)
            tad_results[element] = all([c==tad_value for c in captured_vals])
            tad_matches[element] = [{**m, 'validations':['tad_data_error']} if tad_value!=m['captured_value'] else {**m, 'validations':['tad_pass']} for m in tad_matches[element]] # Determine if tad_data_error is present
        all_match_idxs.extend(match_idxs)

        # USP # EX. "2-0 /&/ 2-0" -> 2-0
        element = 'USP_to' # TO-DO handle cases where they are different
        expression = r'(?<!\d)(\d{1,2}\s*-\s*0)' # Two digits followed by a dash and a 0
        tad_matches[element], match_idxs, captured_vals = self.search_in_text_re(expression, case=False, partial_match=False)
        if self.tad_data:
            tad_value = self.tad_data[element].split(sep)[0].strip() # separate values with " /&/ " in a tuple
            tad_results[element] = all([c==tad_value for c in captured_vals])
            tad_matches[element] = [{**m, 'validations':['tad_data_error']} if tad_value!=m['captured_value'] else {**m, 'validations':['tad_pass']} for m in tad_matches[element]] # Determine if tad_data_error is present
        all_match_idxs.extend(match_idxs)

        # Gauge  # EX. 3 -> (3 Ph. Eur.)
        element = 'gauge_to'# TO-DO handle cases where they are different
        expression = r'(?<!\d)([\d.]{1,3})\s*ph.' # one or two digits (possible decimal point) followed by "ph."
        tad_matches[element], match_idxs, captured_vals = self.search_in_text_re(expression, case=False)
        if self.tad_data:
            tad_value = self.tad_data[element].split(sep)[0].strip() # separate values with " /&/ " in a tuple
            tad_results[element] = all([c==tad_value for c in captured_vals])
            tad_matches[element] = [{**m, 'validations':['tad_data_error']} if tad_value!=m['captured_value'] else {**m, 'validations':['tad_pass']} for m in tad_matches[element]] # Determine if tad_data_error is present
        all_match_idxs.extend(match_idxs)

        # Needle Type # EX. V-5 -> V-5
        element = 'needle_type_to' 
        if not self.tad_data:
            for unique_value in self.process_pdf.vantage_unique[element]:
                tad_matches[element], match_idxs = self.search_in_text(unique_value, partial_match=False)
                # Fetch the relevant row from PS5 since it is tadless and we did not have the needle type from the TAD file
                if match_idxs:
                    if self.get_ps5_data(tad_matches[element][0]['text']):
                        tad_matches[element] = [{**m, 'validations':['ps5_data_pass']} for m in tad_matches[element]]
                    else:
                        tad_matches[element] = [{**m, 'validations':['ps5_data_error']} for m in tad_matches[element]]
        else:
            tad_value = self.tad_data[element] 
            tad_matches[element], match_idxs = self.search_in_text(tad_value)
            tad_results[element] = bool(tad_matches[element])
            tad_matches[element] = [{**m, 'validations':['tad_pass']} for m in tad_matches[element]] # Determine if tad_data_error is present
        all_match_idxs.extend(match_idxs)

        # Needle Length Label # EX. 17 -> "17 mm 1/2c"
        element = 'needle_len_lbl_to'
        expression = r"(?<!x\s)(?<!x)(?<!\d)(\d{1,2}.?\d{1,2})\s?mm(?!\s*x)" # Negative lookarounds to avoid matching "17mm x 12mm" (pledget dimensions)
        # split_bbox: needle_len_lbl_to and curve_frac often live in the same text box
        # (e.g. "17 mm 1/2c"); narrow each match to its own portion of the box instead of
        # both claiming the full (duplicated) box.
        tad_matches[element], match_idxs, captured_vals = self.search_in_text_re(expression, case=False, split_bbox=True)
        if self.tad_data:
            tad_value = str(self.tad_data[element])
            tad_results[element] = all([c==tad_value for c in captured_vals])
            tad_matches[element] = [{**m, 'validations':['tad_data_error']} if tad_value!=m['captured_value'] else {**m, 'validations':['tad_pass']} for m in tad_matches[element]] # Determine if tad_data_error is present
        all_match_idxs.extend(match_idxs)

        # Needle Length L1 # EX. 17.0 -> "17 mm 1/2c"
        # COMMENTED OUT AS IT IS THE SAME AS needle_len_lbl_to
        # element = 'needle_len_l1_to'
        # expression = r"(?<!x\s)(?<!x)(?<!\d)(\d{1,3})\s?mm(?!\s*x)" # Negative lookarounds to avoid matching "17mm x 12mm" (pledget dimensions)
        # tad_matches[element], match_idxs, captured_vals = self.search_in_text_re(expression, case=False)
        # if self.tad_data:
        #     tad_value = str(self.tad_data[element])
        #     tad_results[element] = all([c==tad_value for c in captured_vals])
        # all_match_idxs.extend(match_idxs)
        
        # Needle Curve Degrees # EX. 180 -> Unknown
        # element = 'needle_curve_deg_to'
        # tad_value = str(self.tad_data[element])
        # suffixes = ["°", " deg", " degrees"]
        # tad_matches[element], match_idxs = self.search_in_text(tad_value, suffixes=suffixes, partial_match=True)
        # all_match_idxs.extend(match_idxs)
        # tad_results[element] = bool(tad_matches[element])

        # Needle Curve Fractional # EX. 1/2 -> "17 mm 1/2c"
        element = 'curve_frac'
        expression = r"(?<!\d)(\d/\d)\s?c" # A number, a forward slash, another number, followed by a c
        # split_bbox: share the box with needle_len_lbl_to (see above), so narrow to the
        # "1/2c" portion only.
        tad_matches[element], match_idxs, captured_vals = self.search_in_text_re(expression, case=False, split_bbox=True)
        if self.tad_data:
            if self.ps5_data: 
                tad_value = str(self.ps5_data[element])
            else: 
                tad_value = "Not ps5" # This will simply not match, quick fix in the meantime for tad_validation formally
            tad_results[element] = all([c==tad_value for c in captured_vals])
        all_match_idxs.extend(match_idxs)
        
        # Needle Point # EX. TC -> Symbol
        element = 'needle_point_to'
        if self.tad_data:
            tad_value = self.tad_data[element].strip()
            search = ["NP_"+tad_value]
            tad_matches[element] = self.search_in_symbols('NP_') # Search for all Needle Point symbols
            tad_matches[element] = [{**m, 'validations':['tad_data_error']} if search[0]!=m['symbol_name'] else {**m, 'validations':['tad_pass']} for m in tad_matches[element]] # Determine if tad_data_error is present
            tad_results[element] = all([search[0]==m['symbol_name'] for m in tad_matches[element]]) # All found symbols match TAD

        # Needle point TM # EX. TAPERCUT -> Text OR Literal Symbol
        element = 'needle_point_tm' 
        if not self.tad_data:
            pass
        else:
            tad_value = self.tad_data[element] == "TAPERCUT"
            suffixes = ["TM", " TM", "™", " ™", ""]
            tad_matches[element], match_idxs = self.search_in_text("TAPERCUT", suffixes=suffixes, partial_match=True)
            tad_results[element] = bool(tad_matches[element]) == tad_value
            tad_matches[element] = [{**m, 'validations':['tad_pass']} if tad_value else {**m, 'validations':['tad_data_error']} for m in tad_matches[element]] # Determine if tad_data_error is present
        all_match_idxs.extend(match_idxs)

        # Needle TM # EX. MULTIPASS -> Text OR Literal Symbol
        element = 'needle_tm' 
        if not self.tad_data:
            pass
        else:
            tad_value = self.tad_data[element] == "MULTIPASS"
            suffixes = ["TM", " TM", "™", " ™", ""]
            tad_matches[element], match_idxs = self.search_in_text("MULTIPASS", suffixes=suffixes, partial_match=True)
            tad_results[element] = bool(tad_matches[element]) == tad_value
            tad_matches[element] = [{**m, 'validations':['tad_pass']} if tad_value else {**m, 'validations':['tad_data_error']} for m in tad_matches[element]] # Determine if tad_data_error is present
        all_match_idxs.extend(match_idxs)

        # Needle Alloy # EX. ETHALLOY -> Text OR Literal Symbol
        element = 'needle_alloy_to' 
        if not self.tad_data:
            pass
        else:
            tad_value = self.tad_data[element] == "ETHALLOY"
            suffixes = ["TM", " TM", "™", " ™", ""]
            tad_matches[element], match_idxs = self.search_in_text("ETHALLOY", suffixes=suffixes, partial_match=True)
            tad_results[element] = bool(tad_matches[element]) == tad_value
            tad_matches[element] = [{**m, 'validations':['tad_pass']} if tad_value else {**m, 'validations':['tad_data_error']} for m in tad_matches[element]] # Determine if tad_data_error is present
        all_match_idxs.extend(match_idxs)

        # Sterilization # EX. ETO -> Symbol (STERILE_EO)
        element = 'steri_to'
        if self.tad_data:
            tad_value = self.tad_data[element] in ['EO', 'ETO']
            search = ['STERILE_EO']
            tad_matches[element] = self.search_in_symbols(search)
            tad_results[element] = bool(tad_matches[element]) == tad_value

        # Pieces in box # EX. 6 -> Symbol (N_PC_6)
        # The number of pieces is embedded in the symbol name (N_PC_<n>) by n_pc_post()
        # in yolo_detect (detections_postprocessing), so we read it straight off the symbol
        # instead of searching for a nearby number here.
        element = 'pieces_to'
        symbol_matches = self.search_in_symbols('N_PC')  # matches 'N_PC' and 'N_PC_<n>'
        tad_matches[element] = []
        for m in symbol_matches:
            num = m['symbol_name'].split('_')[-1]
            if not num.isdigit(): # N_PC with no readable number inside; nothing to validate
                continue
            tad_matches[element].append({**m, 'text': num, 'captured_value': num})
        if self.tad_data:
            tad_value = str(self.tad_data[element])
            tad_results[element] = all([m['captured_value']==tad_value for m in tad_matches[element]])
            # Preserve any existing (ACCLM) validations already on the symbol
            tad_matches[element] = [{**m, 'validations': list(m.get('validations', [])) + (['tad_data_error'] if tad_value!=m['captured_value'] else ['tad_pass'])} for m in tad_matches[element]] # Determine if tad_data_error is present            

        # REF number
        # element = 'REF_no'
        # search = ['REF']
        # symbol_matches = self.search_in_symbols(search)
        # matches = []
        # for m in symbol_matches:
        #     el, match_idx = find_next_line(m, self.text_data, thresh=15, mode='right')
        #     if el:
        #         all_match_idxs.extend([match_idx])
        #         matches.extend([el])
        # tad_matches[element] = matches
        # if self.tad_data:
        #     tad_value = str(self.tad_data[element])
        #     tad_results[element] = bool(tad_value==matches[0]['text']) if matches else False

        # LOT number
        element = 'LOT_no'
        search = ['LOT']
        symbol_matches = self.search_in_symbols(search)
        matches = []
        for m in symbol_matches:
            el, match_idx = find_next_line(m, self.text_data, thresh=15, mode='right')
            if el:
                all_match_idxs.extend([match_idx])
                matches.extend([el])
        tad_matches[element] = matches
        if self.tad_data:
            tad_results[element] = True # always True, as there is no TAD entry
        
        # EXPIRY date
        element = 'EXP_date'
        search = ['EXPIRY']
        symbol_matches = self.search_in_symbols(search)
        matches = []
        for m in symbol_matches:
            el, match_idx = find_next_line(m, self.text_data, thresh=15, mode='right')
            if el:
                all_match_idxs.extend([match_idx])
                matches.extend([el])
        tad_matches[element] = matches
        if self.tad_data:
            tad_results[element] = True # always True, as there is no TAD entry

        # MANUFACTURING date
        element = 'MAN_date'
        search = ['MANUFACTURING_DATE']
        symbol_matches = self.search_in_symbols(search)
        matches = []
        for m in symbol_matches:
            el, match_idx = find_next_line(m, self.text_data, thresh=15, mode='right')
            if el:
                all_match_idxs.extend([match_idx])
                matches.extend([el])
        tad_matches[element] = matches
        if self.tad_data:
            tad_results[element] = True # always True, as there is no TAD entry

        # CCN code
        element = 'CCN'
        expression = r"[\dA-Z]{1,15}[.][OPSA][\d]{1,5}"
        tad_matches[element], match_idxs, captured_vals = self.search_in_text_re(expression, case=False)
        if self.tad_data:
            tad_matches[element], match_idxs = self.search_in_text(self.ccn, partial_match=True)
            tad_results[element] = True # always True, as there is no TAD entry
        all_match_idxs.extend(match_idxs)

        # VANTAGE FILE VALIDATIONS

        # Raw material 
        element = 'Raw_Material'
        tad_matches[element] = []
        match_idxs = []
        for unique_value in self.process_pdf.vantage_unique[element]:
            matches, idxs = self.search_in_text(unique_value, partial_match=True)
            matches = [{**m, 'captured_value':m['text']} for m in matches] # Set captured value for comparison later on
            tad_matches[element].extend(matches)
            match_idxs.extend(idxs)
        if self.vantage_data:
            value = self.vantage_data['Scientific Name']
            tad_matches[element] = [{**m, 'validations':['thunderbird_data_error']} if value!=m['text'] and value != 'Not Applicable' else {**m, 'validations':['thunderbird_pass']} for m in tad_matches[element]]
            tad_results[element] = all([m['text']==value for m in tad_matches[element]])
        all_match_idxs.extend(match_idxs)

        # # ETHICON # EX. 17 -> "17 mm 1/2c"
        # element = 'ethicon'
        # expression = r"\s*(ethicon(?:\s*(?:™|®|tm))?)\s*"
        # tad_matches[element], match_idxs, captured_vals = self.search_in_text_re(expression, case=False, partial_match=False)
        # if self.tad_data:
        #     tad_results[element] = True
        # all_match_idxs.extend(match_idxs)

        # Delete matches from self.text_data
        all_match_idxs = set(all_match_idxs)
        for del_i in sorted(all_match_idxs, reverse=True):
            del self.text_data[del_i]

        self.tad_matches = tad_matches
        self.tad_results = tad_results

        #print("Input file element search: ", tad_results)

    def validate_databases(self):
        ps5_cols = { # PS5 : TAD
            'needle_type_to' : 'sales_type_lbl', 
            'needle_len_lbl_to' : 'l1_lbl',
            # 'needle_len_l1_to' : 'l1_std', # Disabled for redundancy
            'needle_curve_deg_to' : 'curve_deg',
            'curve_frac' : 'curve_frac',
        }

        van_cols = { # Vantage : TAD
            "needle_type_to" : "Needle Sales Type",
            #"Product_Name",
            "Raw_Material" : "Scientific Name",
        }

        for element, matches in self.tad_matches.items():
            if element in TAD_COLS.values() and not self.process_pdf.tadless and self.tad_data:
                # tad_data_error is already applied in find_input_elements()
                for match in matches:
                    tad_val = str(self.tad_data[element]).strip()
                    match['expected_value'] = tad_val
            # PS5
            if element in ps5_cols.keys() and self.ps5_data:
                for match in matches:
                    if 'captured_value' in match.keys():
                        cap_val = match['captured_value']
                        ps5_col = ps5_cols[element]
                        ps5_val = str(self.ps5_data[ps5_col]).strip()
                        match['expected_value'] = ps5_val
                        try:
                            ps5_num = float(ps5_val)
                            cap_num = float(cap_val)
                            if abs(ps5_num - cap_num) > 0.02:  # Use small epsilon for float comparison
                                validations_append(match, 'ps5_data_error')
                            else:
                                validations_append(match, 'ps5_pass')
                        except ValueError:
                            # String comparison if values can't be converted to numbers
                            if ps5_val.lower() != cap_val.lower():
                                validations_append(match, 'ps5_data_error')
                            else:
                                validations_append(match, 'ps5_pass')

            # VANTAGE (TAD DUMP)
            if element in van_cols.keys() and self.vantage_data:
                for match in matches:
                    if 'captured_value' in match.keys():
                        cap_val = match['captured_value']
                        van_col = van_cols[element]
                        van_val = str(self.vantage_data[van_col]).strip()
                        match['expected_value'] = van_val
                        try:
                            van_num = float(van_val)
                            cap_num = float(cap_val)
                            if abs(van_num - cap_num) > 0.02:  # Use small epsilon for float comparison
                                validations_append(match, 'thunderbird_data_error')
                            else:
                                validations_append(match, 'thunderbird_pass')
 
                        except ValueError:
                            # String comparison if values can't be converted to numbers
                            if van_val.lower() != van_val.lower():
                                validations_append(match, 'thunderbird_data_error')
                            else:
                                validations_append(match, 'thunderbird_pass')


    def find_brand_name(self):
        """
        find the brand name. If tadless just take from the match to the database. If not, look for all brands in the page
        """
        if self.tad_data:
            self.process_pdf.brand_name = self.tad_data['brand']
        else:
            brand_matches = self.tad_matches.get('brand', [])
            if brand_matches:
                self.process_pdf.brand_name = brand_matches[0]['text']

    def search_in_symbols(self, search):
        """
        Search for symbols in symbol_data that either:
        - Start with the search term (if search is a string)
        - Match any item in the search list (if search is a list)

        Args:
            search: String or list of strings to search for

        Returns:
            list: List of matching symbols from self.symbol_data
        """
        matches = []
        
        if not self.symbol_data:
            return matches
            
        # Handle string search
        if isinstance(search, str):
            for symbol in self.symbol_data:
                if symbol["symbol_name"].startswith(search):
                    matches.append(symbol)
        
        # Handle list search
        elif isinstance(search, (list, tuple)):
            for symbol in self.symbol_data:
                if symbol["symbol_name"] in search:
                    matches.append(symbol)
        return matches

    def search_in_text(self, value, prefixes=None, suffixes=None, match_case=False, partial_match=False, multibox=False):
        """
        Search for text elements in the label that match the given value with optional prefixes/suffixes.
        
        Args:
            value (str): The text value to search for
            prefixes (list): Optional list of prefixes to match before the value
            suffixes (list): Optional list of suffixes to match after the value
            match_case (bool): Whether to perform case-sensitive matching
            
        Returns:
            list: List of matching text elements from self.text_data
        """
        prefixes=[""] if prefixes==None else prefixes
        suffixes=[""] if suffixes==None else suffixes
        matches = []
        match_idxs = []

        #value = value.replace("’", "'") # Standardize apostrophes
                
        # Convert value to lowercase if case-insensitive
        if not match_case:
            value = value.lower()
            prefixes = [p.lower() for p in prefixes]
            suffixes = [s.lower() for s in suffixes]
            
        # Build all possible combinations
        patterns = [f"{prefix}{value}{suffix}" for prefix in prefixes for suffix in suffixes]
        
        # Search through text elements
        for i, element in enumerate(self.text_data):
            text = element["text"] if match_case else element["text"].lower()
            text = text.strip()
            #text = text.replace("’", "'") # Standardize apostrophes
            if partial_match:
                if any(pattern in text for pattern in patterns):
                    matches.append(element)
                    match_idxs.append(i)
            else:
                if any(text == pattern for pattern in patterns):
                    matches.append(element)
                    match_idxs.append(i)
                elif any(text in pattern for pattern in patterns) and multibox:
                    for pattern in patterns:
                        start_idx = pattern.find(text)
                        remaining_text = pattern[:start_idx] + pattern[start_idx+len(text):]
                        matches2, idxs2 = self.search_in_text(remaining_text.strip())
                        if matches2 and bbox_distance(element['bbox'], matches2[0]['bbox'], min_orthogonal=True) < 10:
                            merged = merge_elements(matches2[0], element)
                            matches.append(merged)
                            match_idxs.append(i)
                            match_idxs.append(idxs2[0]) # Remaining text
                    
        return matches, match_idxs

    def search_in_text_re(self, expression, case=True, partial_match=True, split_bbox=False):
        """
        Search for text elements in the label that match the regular expression
        Captured groups will be returned if the expression has any (will return only first one)

        Args:
            expression: RE to compile
            case: If false al texts are lowercased before the re is searched for
            partial_match: If false it will not match unless the match is the entire text box (ignoring 
                           leading/trailing space). Lookaheads and lookbehinds could alter this behaviour.
            split_bbox: If True, the returned match bbox is narrowed to just the matched
                        substring instead of spanning the whole text box. The sub-bbox is
                        estimated assuming a monospace font (char width = box width / len(text))
                        and respecting the text orientation. This lets several elements that
                        share one box (e.g. "17 mm 1/2c" -> needle_len_lbl_to + curve_frac)
                        each get their own, non-overlapping bbox.
        """
        matches = []
        match_idxs = []
        captured = []
        
        pattern = re.compile(expression)
                      
        # Search through text elements
        for i, element in enumerate(self.text_data):
            element = element.copy() # Avoid multiple matches to same text box from overwriting capture_value
            orig_text = element["text"].strip() # Original-case stripped text (for slicing out the match)
            text = orig_text
            if not case:
                text = text.lower()
            match = pattern.search(text)
            if match:
                if partial_match or (len(text)==len(match.group(0))):
                    try: # get the capture groups if there are any
                        cap = match.group(1)
                    except IndexError:
                        cap = ''
                    captured.append(cap)
                    element['captured_value'] = cap
                    if split_bbox and len(text) > 0:
                        element['bbox'] = self._split_bbox_monospace(
                            element['bbox'], len(text), match.start(), match.end(),
                            element.get('orientation', 0))
                        element['center'] = bbox_center(element['bbox'])
                        # Keep only the matched portion in 'text' so this element
                        # carries just its relevant substring (e.g. "17 mm" or "1/2c")
                        # instead of the full shared box. Slice from the original-case
                        # text so casing is preserved.
                        element['text'] = orig_text[match.start():match.end()].strip()
                    matches.append(element)
                    match_idxs.append(i)
        return matches, match_idxs, captured

    @staticmethod
    def _split_bbox_monospace(bbox, n_chars, start, end, orientation):
        """
        Estimate the bbox of a substring [start:end] within a text box, assuming a
        monospace font so each character takes an equal share of the box along the
        text's reading direction. The orientation selects the axis and direction the
        text advances along:
            0   -> left to right along x
            180 -> right to left along x
            90  -> top to bottom along y
            270 -> bottom to top along y
        """
        x0, y0, x1, y1 = bbox
        if orientation in (90, 270):
            char = (y1 - y0) / n_chars
            if orientation == 90:
                ny0, ny1 = y0 + start * char, y0 + end * char
            else:  # 270, first char at the bottom
                ny0, ny1 = y1 - end * char, y1 - start * char
            return [x0, int(round(ny0)), x1, int(round(ny1))]
        else:
            char = (x1 - x0) / n_chars
            if orientation == 180:  # first char at the right
                nx0, nx1 = x1 - end * char, x1 - start * char
            else:  # 0
                nx0, nx1 = x0 + start * char, x0 + end * char
            return [int(round(nx0)), y0, int(round(nx1)), y1]

    def find_translations(self):
        """
        Format 1
        |en English Text|

        Format 2
        |en-|English Text /|

        Format 3
        |en|-English Text /|
        
        """
        translation_data = []
        all_match_idxs = []
        font_offender_idxs = set() # Indices of individual boxes that fail the font-size check
        for i, element in enumerate(self.text_data):
            if element["text"].startswith('en') or element["text"].startswith('en-'):
                dashed_mode = element["text"].startswith('en-') or \
                                element["text"] == "en"  # This format has lan and trans in separate boxes
                en_text = element["text"].replace("en ", "").strip() # Get the english text for looking up in the db
                if dashed_mode: # Get the look-up text from the contiguous bbox
                    en_text_element, _ = find_next_line(element, self.text_data, thresh=15, mode='right')
                    if not en_text_element:
                        continue # No second line found
                    # First two replace gets rid of a weird /xa0 and \x03 character, third gets rid of forward slash
                    en_text = en_text_element["text"].replace(u'\xa0', u' ').replace('/', "") 
                    if en_text.startswith("-"):
                        en_text = en_text[1:] # Get rid of dash at beginning of en text
                for df_name, df in self.process_pdf.translation_database.items():
                    #Search for en_text in the "en" column of the df and get the entire row
                    matches = df[df['en'].str.lower() == en_text.strip().lower()]
                    if not matches.empty:
                        translation_row = matches.iloc[0].to_dict()
                        translation_item = {'type':df_name, 
                                            'name':translation_row[df_name], 
                                            'bbox':[np.inf, np.inf, -np.inf, -np.inf], 
                                            'validations' : []}
                        item_match_idxs = [] # Indices of every text box that makes up this block, for the font check
                        extra_elements = [] # Boxes that are part of the block but not tracked by index
                        if dashed_mode and en_text_element:
                            # The english look-up box is part of the block too, include it in the font check
                            extra_elements.append(en_text_element)
                        for n, (lan, trans) in enumerate(translation_row.items()):
                            if n!=0: #Skip first one as it is the type and name, not a translation
                                search = lan + '    ' + trans
                                if lan=="ja": # Japanese text special case. lan and trans are in different boxes
                                    search = trans
                                    _, match_idx = self.search_in_text('ja') # find 'ja' just to delete it
                                    all_match_idxs.extend(match_idx)
                                    item_match_idxs.extend(match_idx)
                                if dashed_mode: # Dashed mode, trans and lan are on separatee boxes
                                    search = trans
                                    _, match_idx_lan = self.search_in_text(lan,
                                                                           suffixes=['-',''], 
                                                                           prefixes=['', '/', '/ '],
                                                                           match_case=True) # find lan box 
                                    all_match_idxs.extend(match_idx_lan) 
                                    item_match_idxs.extend(match_idx_lan)
                                    if not match_idx_lan: # flag missing lan markers
                                        translation_item['validations'].append('lan_'+lan+'_error')
                                suffixes = ['', '/', ' /', u'\xa0/'] # languages might be separated by a slash (and/or have that weird \xa0 char)
                                prefixes = ['', '-', '- ', '/ ', '/'] # texts might start with a dash
                                match, match_idx = self.search_in_text(search.strip(), prefixes=prefixes, suffixes=suffixes, multibox=True)
                                all_match_idxs.extend(match_idx)
                                item_match_idxs.extend(match_idx)
                                if match:
                                    # Expand bbox to include match
                                    match = match[0]
                                    translation_item['bbox'] = expand_bboxes(translation_item['bbox'], match['bbox'])
                                else:
                                    translation_item['validations'].append('lan_'+lan+'_error')
                                if dashed_mode and match and match_idx_lan: # Check that lna and text are next to each other
                                    if not any([is_next_line(self.text_data[m_lan], self.text_data[m], mode='right') for m in match_idx for m_lan in match_idx_lan]): # This is a loop for the case where the  lan or the text has multiple boxes, we want any pair to be contiguaous
                                        if lan in ['ar', 'he']: # check reverse order for right-to-left languages
                                            if not any([is_next_line(self.text_data[m], self.text_data[m_lan], mode='right') for m in match_idx for m_lan in match_idx_lan]):
                                                translation_item['validations'].append('lan_'+lan+'_error')
                                        else:
                                            translation_item['validations'].append('lan_'+lan+'_error')
                        translation_item['match'] = not bool(translation_item['validations'])
                        if translation_item['match']:
                            validations_append(translation_item, 'SEA_pass')
                        else:
                            validations_append(translation_item, 'SEA_data_error')
                            translation_item['validations'] = list(set(translation_item['validations'])) # Remove duplicates
                        # Font size check: the block is made of several boxes that may use different
                        # fonts (bold language indicators, arabic script, etc.) so there is no single
                        # font to compare. Reuse font_check per box and flag the whole block if any
                        # constituent box triggers a fontsize_small_error. In addition, remember which
                        # individual boxes are the offenders so they can be kept in text_data and
                        # flagged individually (helps users pinpoint the offending language/text).
                        block_offender_idxs = [m for m in item_match_idxs
                                               if font_check_block([self.text_data[m]], self.process_pdf.document_type)]
                        if block_offender_idxs or font_check_block(extra_elements, self.process_pdf.document_type):
                            validations_append(translation_item, 'fontsize_small_error')
                        font_offender_idxs.update(block_offender_idxs)
                        translation_data.append(translation_item)
                    else:
                        # No match found in translation file
                        pass
        
        all_match_idxs = set(all_match_idxs)
        # Keep the individual boxes that fail the font-size rule (flag them instead of
        # deleting) so users can locate exactly which text/language is too small.
        for offender_i in font_offender_idxs:
            validations_append(self.text_data[offender_i], 'fontsize_small_error')
        delete_idxs = all_match_idxs - font_offender_idxs
        for del_i in sorted(delete_idxs, reverse=True):
            del self.text_data[del_i]

        # Look for all boxes inside translation items, and add custom error so that they remain but are flagged
        for translation_item in translation_data:
            for i, element in enumerate(self.text_data):
                if bbox_inside_bbox_soft(element['bbox'], translation_item['bbox']):
                    if 'fontsize_small_error' in element.get('validations', []):
                        continue # Already kept and flagged as a font-size offender
                    validations_append(element, 'translation_error')

        self.translation_data = translation_data
    
    def find_addresses(self):
        """
        Localize address elements in the labels and verify that they match exactly to the database.
        It works by iterating over the text data, when it finds something that matches the 
        beginning of one of the known addresses, it will look for the next line (the one 'below', 
        considering the orientation of the text) and append it to the found address if its text is 
        contained in that db address. When the full address is extracted it is compared to match 
        exactly with the one in the database.
        """
        def append_next_address_line(curr_element, one_line):
            for i, r_el in enumerate(self.text_data):
                if is_next_line(curr_element, r_el, thresh=80, gap_thresh=30): # Increased thres2 h from 10 to 80 to match unaligned lines from manufacturer symbol presence
                    clean_text = r_el['text'].lower().replace(',', '').replace('.', '').replace(' •', '').strip()
                    clean_text = re.sub(r'\s+', ' ', clean_text) # Clean double spaces too
                    if clean_text in one_line and len(clean_text) > 2: # Ignore shrt matches such as "es" "et" that create false positives
                        one_line = one_line.replace(clean_text, "", 1) # Remove to avoid recursion loop
                        next_text, next_box = append_next_address_line(r_el, one_line)
                        all_match_idxs.extend([i])
                        return [curr_element['text'].strip()] + next_text, expand_bboxes(curr_element['bbox'], next_box)
            return [curr_element['text'].strip()], curr_element['bbox']

        address_data = []
        all_match_idxs = []

        for i, element in enumerate(self.text_data):
            for name, address in self.process_pdf.manufacturer_address_db.items():
                one_line = ' '.join(address).replace(',', '').replace('.', '').lower() # Address as one line and without commas
                element_text_clean = element['text'].lower().replace(',', '').replace('.', '').replace('•', '').strip() 
                element_text_clean = re.sub(r'\s+', ' ', element_text_clean) # Clean double spaces too
                if one_line.startswith(element_text_clean):
                    one_line = one_line.replace(element_text_clean, "", 1)
                    start_len = len(all_match_idxs) # Track the boxes that make up this address for the font check
                    address_found, address_bbox = append_next_address_line(element, one_line)
                    if len(address_found) > 1: # Not only went to depth one (filter matches to only first line)
                        item_match_idxs = list(all_match_idxs[start_len:]) + [i] # Following lines + the starting box
                        all_match_idxs.extend([i])
                        validations = []
                        if not address_found == address: # This checks for identical match, including line breaks
                            # This checks if cleaning, punctuation, line breaks and case yields a match, if yes, isolate each case
                            if ' '.join(address_found).replace(',','').replace('.','').lower() == ' '.join(address).replace(',','').replace('.','').lower():
                                if [l.lower().replace(',','').replace('.','') for l in address_found] != [l.lower().replace(',','').replace('.','') for l in address]:
                                    # validations.append('address_line_break_error')
                                    pass
                                if ' '.join(address_found).replace(',','').replace('.','') != ' '.join(address).replace(',','').replace('.',''):
                                    validations.append('address_case_error')
                                if ' '.join(address_found).lower() != ' '.join(address).lower():
                                    # validations.append('address_punctuation_error')
                                    pass
                            # Check for doubled spaces
                            elif re.sub(r'\s+', ' ', ' '.join(address_found)).strip() == re.sub(r'\s+', ' ', ' '.join(address)).strip():
                                validations.append('address_space_error')
                            else: # Difference not due to punctuation, case or line breaks, append generic error
                                validations.append('address_error')
                        if validations:
                            validations.append('address_db_data_error') # Append data error to show the DB in parenthesis
                        else:
                            validations.append('address_db_pass')
                        # Font size check: an address block is assembled from several boxes that may
                        # use different fonts. Reuse font_check per box and flag the whole block if
                        # any constituent box triggers a fontsize_small_error.
                        block_elements = [self.text_data[m] for m in item_match_idxs]
                        if font_check_block(block_elements, self.process_pdf.document_type):
                            validations.append('fontsize_small_error')
                        address_data.append({#'address' : " ".join(address),
                                                  'name' : name,
                                                  'validations' : validations,
                                                  'bbox' : address_bbox,
                                                  # Carry the text orientation so downstream
                                                  # checks (e.g. required-symbol placement) know
                                                  # which page direction is the reading-"left".
                                                  'orientation' : element.get('orientation', 0)})
        
        # Delete individual address lines from data
        all_match_idxs = set(all_match_idxs)
        for del_i in sorted(all_match_idxs, reverse=True):
            del self.text_data[del_i]
                        
        self.address_data = address_data

    def decode_barcodes(self):
        """
        decode BARCODE and DATAMATRIX detections from the object detection model.
        The page is re-rendered at a higher resolution so codes are decoded/graded
        from a sharper image. The bbox and pixels_to_mm are scaled accordingly, and
        detected bboxes are scaled back to original image coordinates.
        """
        remove_sym = []
        scale = 2  # Upscale factor for the barcode pixmap
        pixels_to_mm = self.pixels_to_cm * 10 / scale  # mm/pixel adjusted for the upscaled image
        hires_image = None  # Rendered lazily, only if a code is present
        # Only save barcode crops for the 'to' (future) label
        barcode_id=1
        for i, symbol in enumerate(self.symbol_data):
            if symbol['symbol_name'] in ['BARCODE', 'DATAMATRIX']:
                remove_sym.append(i)
                pad = 20 #padding
                bbox = symbol['bbox']

                # Re-render the page once at higher resolution
                if hires_image is None:
                    hires_mat = self.mat * fitz.Matrix(scale, scale)
                    hires_pix = self.page.get_pixmap(matrix=hires_mat)
                    hires_image = np.frombuffer(hires_pix.samples, dtype=np.uint8).reshape(hires_pix.height, hires_pix.width, hires_pix.n)
                    hires_image = cv2.cvtColor(hires_image, cv2.COLOR_RGB2BGR)

                # Scale the bbox to the high-resolution image space
                hires_bbox = [int(round(coord * scale)) for coord in bbox]

                # The HRI check runs inside the detector; text_data is in original
                # image space, so pass the hires->original scale and a neighbor
                # distance threshold (~0.1 cm) in that original space.
                detected_codes = detect_barcode_datamatrix(
                    hires_image, bbox=hires_bbox, btype=symbol['symbol_name'].lower(),
                    pixels_to_mm=pixels_to_mm, text_data=self.text_data, text_scale=scale,
                    hri_distance_threshold=.1 / self.pixels_to_cm)
                for code in detected_codes:
                    # Scale detected bbox back to original image coordinates
                    code['bbox'] = [int(round(coord / scale)) for coord in code['bbox']]
                    crop_filename = f"page_{self.page_num + 1}_barcode_{barcode_id}.png"
                    barcode_id += 1
                    if self.page_type == "to":
                        save_barcode_crop(self.images['original'], code['bbox'], self.process_pdf.output_folder, filename=crop_filename,
                                            mode=self.process_pdf.mode)
                        code['crop_filename'] = crop_filename

                self.barcodes.extend(detected_codes)

        for i in remove_sym[::-1]:
            self.symbol_data.pop(i)

    def validate_barcodes(self):
        """
        Check that barcodes don't have any leading/trailing spaces or non-printabe characters.
        The HRI (human-readable text) check is performed inside
        ``detect_barcode_datamatrix`` (reported as the 'HRI' grade metric); its
        result is pulled here into the validations so it surfaces as before.
        """
        for barcode in self.barcodes:
            decoded = barcode['decoded']
            if decoded:
                decoded_stripped = decoded.strip()
                decoded_stripped = ''.join(filter(str.isprintable, decoded_stripped))
                if decoded != decoded_stripped:
                    validations_append(barcode, 'space_error')
            else:
                validations_append(barcode, 'decode_error')
            # Pull the HRI check result computed during decoding. HRI_found is None
            # when no check ran (e.g. decode failure or no text_data).
            hri_found = barcode.get('HRI_found')
            if hri_found is True:
                validations_append(barcode, 'HRI_pass')  # HRI (human-readable text) matches the decoded value
            elif hri_found is False:
                validations_append(barcode, 'HRI_data_error')  # HRI (human-readable text) does not match the decoded value

    def get_eurocode(self):
        eurocode_pattern = re.compile(r"[\dA-Z]{1,15}[.][OPSA][\d]{1,5}")
        self.ccn = ''
        for word in self.text_data:
            match = eurocode_pattern.search(word["text"])
            if match:
                self.ccn = match.group(0)
                break
        if not self.ccn:
            pass
            #raise PageError(f"No Eurocode found in the {self.page_type} page")

    def get_code_data(self):
        target_eurocode = self.ccn.split(".")[0]  # Remove any suffix after a dot
        
        if not self.process_pdf.tadless:
            if target_eurocode:
                # Fetch the matching row from df_tad
                matching_row = self.process_pdf.df_tad[self.process_pdf.df_tad["code_to"] == target_eurocode]
                if not matching_row.empty:
                    self.tad_data = matching_row.iloc[0].to_dict()
                    print(f"Found matching TAD data for code: {target_eurocode}")
                else:
                    # try to match without last character
                    matching_row = self.process_pdf.df_tad[self.process_pdf.df_tad["code_to"].str[:-1] == target_eurocode]
                    if not matching_row.empty:
                        self.tad_data = matching_row.iloc[0].to_dict()
                        print(f"Found matching TAD data for code: {target_eurocode}")
                    else:
                        print(f"No matching TAD data found for code: {target_eurocode}")
                        # raise PageError(f"No matching TAD data found for code: {target_eurocode}")

            else:
                raise PageError("No Eurocode found in the page")

            # Get needle information from PS-5
            if self.tad_data:
                self.get_ps5_data(self.tad_data['needle_type_to'])

        # Get matching row from Vantage file
        if target_eurocode:
            # Fetch the matching row from df_vantage
            matching_row = self.process_pdf.df_vantage[self.process_pdf.df_vantage["EURO CODE"] == target_eurocode]
            if not matching_row.empty:
                self.vantage_data = matching_row.iloc[0].to_dict()
                print(f"Found matching Vantage data for code: {target_eurocode}")
            else:
                # try to match without last character
                matching_row = self.process_pdf.df_vantage[self.process_pdf.df_vantage["EURO CODE"].str[:-1] == target_eurocode]
                if not matching_row.empty:
                    self.vantage_data = matching_row.iloc[0].to_dict()
                    print(f"Found matching Vantage data for code: {target_eurocode}")
                else:
                    # raise PageError(f"No matching Vantage data found for code: {target_eurocode}")
                    pass
        else:
            print("No Eurocode found in the page")

    def get_ps5_data(self, needle_type_to):
        filtered_ps5 = self.process_pdf.df_ps5[self.process_pdf.df_ps5["sales_type_lbl"].str.strip().str.lower() == needle_type_to.strip().lower()]
        if not filtered_ps5.empty:
            self.ps5_data = filtered_ps5.iloc[0].to_dict()
            return True
        else:
            # raise PageError(f"No PS-5 data found for needle type: {needle_type_to}")
            return False

    def symbols_yolo(self, ignore_symbols=None, acclm=False):
        """
        Use the loaded yolo model to run inference on the image and find symbols

        Args:
            ignore_symbols: Optional list of class names to discard (symbols that never
                appear in this business unit's labels, e.g. ['NEEDLE'] for EP/VISION).
            acclm: When True, validate detected symbols against the ACCLM library. Only
                SURGERY enables this; EP and VISION leave it off.
        """
        ignore_symbols = ignore_symbols or []

        # Brand logos: OCR text matching a configured trigger phrase (e.g. 'Biosense Webster',
        # 'Johnson&Johnson') triggers a template match of the logo artwork in data/logos.
        # Confirmed logos are added to symbol_data and the OCR text they replace is removed.
        logo_detection(self.images['original'], self.text_data, self.symbol_data)

        def symbol_orientation(det):
            """
            Get the orientation of the detected symbol by taking the stock symbol and calculating overlap (template matching)
            over all orientations and multiple scales. The best overlap is the orientation.
            """
            box = det['bbox']
            pad = 20
            crop = self.images['no_text'][max(0, box[1]-pad):box[3]+pad, max(0, box[0]-pad):box[2]+pad]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            template_path = os.path.join(str(ROOT_DIR) + "/data/object_detection_symbols", f"{det['class_name']}.png")
            template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
            
            scale_ratio = (crop.shape[0] -pad*2 + crop.shape[1] -pad*2) /  (template.shape[0] + template.shape[1])
            template = cv2.resize(template, (0,0), fx = scale_ratio, fy = scale_ratio) # Scale to roughly detection size

            tm_mode = cv2.TM_CCOEFF_NORMED

            orientations = [0, 90, 180, 270]
            scales = [.8, .9, 1.0, 1.1, 1.2] 
            best_score = 0
            best_orientation = 0 # default
            for i in range(4):
                if i>0:
                    template = cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE)
                for scale in scales:
                    scaled_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
                    if crop.shape[0] < scaled_template.shape[0] or crop.shape[1] < scaled_template.shape[1]: 
                        # rare case when scale is large can make the template larger than the crop
                        continue
                    result = cv2.matchTemplate(~crop, ~scaled_template, tm_mode )
                    score = result.max()
                    if score > best_score:
                        best_score = score
                        best_orientation = orientations[i]
            return best_orientation
        
        def needle_symbol_check(detection):
            """
            This checks if a Needle detection isn't actually just a needle symbol (the one used to 
            indicate number of needles)
            It looks for something like "8 x" left of the detection.
            """
            expression = r"(?:\d{1,2}\s*x|x\s*\d{1,2})"# A number and an x, in any order or with spaces
            text_matches, _, captured_vals = self.search_in_text_re(expression, case=False, partial_match=False)
            for m in text_matches:
                # re-use is_next_line to determine if the symbol is next to the "8 x" text
                if is_next_line(m, {'bbox':detection['bbox']}, thresh=25, mode='right'):
                    return True
            return False

        # Classes where orientation needs to be determined because of mirrored classes
        # The second class is the mirrored version of the first
        # First class is what is detected by the model, class can stay the same or change, depending on orientation
        reversible_needle_points = {'NP_CEC':'NP_CER',
                                'NP_CEPC':'NP_CEPR',
                                'NP_CESC':'NP_CESR',
                                'NP_CEUC':'NP_CEUR'}
        if self.upper_right_bbox: # Multilabel, only run upper right label
            urbb = self.upper_right_bbox
            inference_image = self.images["no_text"][urbb[1]:urbb[3], urbb[0]:urbb[2], :]
        else:
            inference_image = self.images["no_text"]

        h, w = inference_image.shape[:2]
        quad_thresh = 2500
        if h>quad_thresh or w>quad_thresh:
            mode = 'quad' #Image is to large, split into 4 subimages and do inference
        else:
            mode = 'normal'
        inference_image = cv2.cvtColor(inference_image, cv2.COLOR_BGR2GRAY) # Convert to grayscale
        # _, inference_image = cv2.threshold(inference_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Binarize Otsu
        # inference_image = cv2.cvtColor(inference_image, cv2.COLOR_GRAY2BGR) # Keep 3 channels
        # inference_image = cv2.adaptiveThreshold(inference_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) # Binarize (takes grayscale)
        inference_image = cv2.merge([inference_image, inference_image, inference_image]) # Keep 3 channels
        
        detections = detect_objects(self.process_pdf.yolo_model, inference_image, mode = mode, 
                                    conf_thresh = 0.6,
                                    iou_threshold=0.60)
        if self.upper_right_bbox: # Correct bboxes to original coordinates
            for det in detections:
                det['bbox'] = [det['bbox'][0] + urbb[0], 
                              det['bbox'][1] + urbb[1], 
                              det['bbox'][2] + urbb[0], 
                              det['bbox'][3] + urbb[1]]

        detections = box_symbol_detection(self.images["no_text"], self.text_data, detections)
        # Dark/colour-filled box symbols (white text on a solid colour box, e.g. NAV, 8mm).
        # Uses the original image so the white glyphs are still present.
        detections = dark_box_symbol_detection(self.images["original"], self.text_data, detections)
        # Circle symbols (white text inside a solid colour-filled circle, e.g. curve-type D, F, XR).
        # Uses the original image so the white glyphs are still present.
        detections = circle_symbol_detection(self.images["original"], self.text_data, detections)
        # Strand-count symbols (CR/n, MS/n) read verbatim by OCR and confirmed by template matching.
        detections = strand_symbol_detection(self.images["original"], self.text_data, detections)
        detections = detections_postprocessing(detections, self.images["original"], self.text_data, acclm=acclm)
        # Drop symbols that don't exist in this business unit's labels
        if ignore_symbols:
            detections = [d for d in detections if d["class_name"] not in ignore_symbols]
        detections = sorted(detections, key=lambda x: x["class_name"] == "NEEDLE") # Move NEEDLE to the back (to get the crops without symbols)
        for det in detections:
            class_name = det['class_name']
            object_type = class_name.split('_')[0] #NP, PLED or NEEDLE

            if object_type == 'NEEDLE':
                if needle_symbol_check(det):
                    object_type = 'NEEDLES' #NEEDLES is different to NEEDLE
                    class_name = 'NEEDLES' #NEEDLES is the symbol indicating needle count, NEEDLE is a needle profile
            if object_type == 'NEEDLE':
                xmin, ymin, xmax, ymax = det['bbox']
                needle_crop = self.images["no_text"][ymin:ymax, xmin:xmax]
                self.needle_data.append({
                        'bbox': det['bbox'],
                        'center':  bbox_center(det['bbox']),
                        'confidence': det['confidence'],
                        'crop': needle_crop
                        })      
            else:        
                if object_type in ['NP', 'PLED'] and class_name in reversible_needle_points.keys():
                    # Determine symbol orientation
                    symbol_o = symbol_orientation(det)
                    # Check orientation of neighboring text
                    text_o = orientation_neighbors(bbox_center(det['bbox']), self.text_data)
                    
                    # Determine what symbol it is
                    if symbol_o!=text_o:
                        class_name = reversible_needle_points[class_name]  
                if class_name == 'BOX_SYM':
                    inside_text = find_element_inside_bbox(det['bbox'], self.text_data)
                    if inside_text:
                        letters_only = re.sub(r"[^a-zA-Z]", "", inside_text['text']).upper() # Keep only letters
                        for sym in BOX_SYMBOLS:
                            if letters_only == sym.replace('_',''): # Remove underscores
                                class_name = sym
                            else:
                                continue # Do not add BOX_SYM as a symbol if OCR text is not in BOX_SYMBOLS
                    else:
                        continue # Do not add BOX_SYM as a symbol if there is no OCR text
                bbox = det['bbox']
                sym_color, bg_color = get_text_color(self.images["no_text"][bbox[1]:bbox[3], bbox[0]:bbox[2]])
                # Dark-box symbols carry their extracted fill colour; keep it as the symbol colour.
                if det.get('color') is not None:
                    sym_color = det['color']
                match = {
                        'symbol_name': class_name,
                        'symbol_type': "Image",
                        'bbox': bbox,
                        'center':  bbox_center(det['bbox']),
                        'confidence': det['confidence'],
                        'size': 1,
                        'font': 'Image_Symbol',
                        'color': sym_color,
                        'validations': list(det.get('validations', [])), # ACCLM validation (acclm_pass/acclm_data_error) if present
                            }
                self.symbol_data.append(match)
                # Remove symbol from image (for better needle comparison)
                # background_color = get_common_colors(self.images["no_text"][bbox[1]:bbox[3], bbox[0]:bbox[2]], 
                #                                      white_threshold=255, black_threshold=0) # Include black and white
                h, w, _ = self.images["no_text"][bbox[1]:bbox[3], bbox[0]:bbox[2]].shape
                if bg_color:
                    self.images["no_text"][bbox[1]:bbox[3], bbox[0]:bbox[2]] = np.full((h, w, 3), bg_color, dtype=np.uint8)         


        # --- ACCLM validation for special-font symbols -----------------------------------
        # YOLO/image symbols are validated against the ACCLM library inside
        # detections_postprocessing(); the font-decoded symbols added in read_page() are
        # not. Run the same check on them here so they get validated too. The crop MUST be
        # taken from images['original']: font glyphs are redacted out of images['no_text'],
        # so a 'no_text' crop would be blank. Only when ACCLM validation is enabled.
        if acclm:
            for sym in self.symbol_data:
                if sym.get('font') == 'Image_Symbol':
                    continue  # image symbols were already validated during YOLO post-processing
                acclm_validation = acclm_validate_symbol(sym['symbol_name'], sym['bbox'], self.images['original'])
                if acclm_validation:
                    sym.setdefault('validations', []).append(acclm_validation)

        # --- Deduplicate font-decoded vs YOLO/image detections of the same symbol ----------
        # A symbol drawn with an Ethicon symbol font is added in read_page(); its glyph is
        # only redacted with a thin line, so YOLO (and box_symbol_detection) can re-detect the
        # same mark as an 'Image_Symbol', producing a duplicate at the same location (e.g. the
        # hourglass "Use by date" or box symbols like LOT/STERILE_EO). Drop the redundant image
        # detection, keeping the font-decoded one whose name is used downstream by
        # find_input_elements() (LOT/REF/EXPIRY next-line extraction).
        def _overlap_frac(b1, b2):
            ix = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
            iy = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
            inter = ix * iy
            min_area = min((b1[2] - b1[0]) * (b1[3] - b1[1]), (b2[2] - b2[0]) * (b2[3] - b2[1]))
            return inter / min_area if min_area > 0 else 0

        remove_idx = []
        for i, img_sym in enumerate(self.symbol_data):
            if img_sym.get('font') != 'Image_Symbol':
                continue
            for font_sym in self.symbol_data:
                if font_sym is img_sym or font_sym.get('font') == 'Image_Symbol':
                    continue
                if _overlap_frac(img_sym['bbox'], font_sym['bbox']) > 0.5:
                    # The image detection sometimes resolves a more specific variant of the
                    # same base symbol than the font decoder did (e.g. font 'CE' vs image
                    # 'CE_2797_H' from ce_post, or 'RX' vs 'RX_ONLY'). Carry that
                    # specialization - name plus the wider bbox that now includes the
                    # notified-body number / 'only' text - onto the retained font symbol
                    # before dropping the image duplicate.
                    img_name = img_sym.get('symbol_name', '') or ''
                    font_name = font_sym.get('symbol_name', '') or ''
                    if font_name and img_name.startswith(font_name + '_'):
                        font_sym['symbol_name'] = img_name
                        font_sym['bbox'] = expand_bboxes(font_sym['bbox'], img_sym['bbox'])
                        font_sym['center'] = bbox_center(font_sym['bbox'])
                    remove_idx.append(i)  # same physical symbol already captured by the font decoder
                    break
        for i in sorted(set(remove_idx), reverse=True):
            self.symbol_data.pop(i)

        # --- Deduplicate overlapping image detections of the same physical mark -------------
        # An OCR-confirmed box/circle detection (confidence 1.0) can coexist with the raw
        # YOLO detection of the same mark when the box detection failed to replace it (e.g.
        # STERILE_EO/UDI detected twice, or a spurious YOLO class like '69 - REMOVE' sitting
        # on top of the real 'ELECTRODES_EA' box). When two image symbols overlap heavily,
        # keep the higher-confidence one (box/circle detections carry confidence 1.0, so the
        # OCR-validated symbol wins over a lower-confidence YOLO guess).
        dup_remove = set()
        for i in range(len(self.symbol_data)):
            if i in dup_remove:
                continue
            si = self.symbol_data[i]
            if si.get('font') != 'Image_Symbol':
                continue
            for j in range(i + 1, len(self.symbol_data)):
                if j in dup_remove:
                    continue
                sj = self.symbol_data[j]
                if sj.get('font') != 'Image_Symbol':
                    continue
                if _overlap_frac(si['bbox'], sj['bbox']) > 0.7:
                    # Drop the lower-confidence detection (ties keep the earlier one).
                    if sj.get('confidence', 0) > si.get('confidence', 0):
                        dup_remove.add(i)
                        break
                    else:
                        dup_remove.add(j)
        for i in sorted(dup_remove, reverse=True):
            self.symbol_data.pop(i)

        # --- Minimum-size check for regulatory conformity marks ----------------------------
        # CE, UKCA and the Ukraine "UA Conformity Sign" must be at least
        # MIN_SYMBOL_SIZE_MM wide and tall. Flag any that fall short with a
        # 'symbol_size_error' so it surfaces through the normal validations pipeline.
        min_size_px = MIN_SYMBOL_SIZE_MM / 10 / self.pixels_to_cm  # mm -> cm -> pixels
        for sym in self.symbol_data:
            name = sym.get('symbol_name', '') or ''
            if any(name.startswith(prefix) for prefix in SIZE_CONSTRAINED_SYMBOL_PREFIXES):
                bbox = sym['bbox']
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                if width < min_size_px or height < min_size_px:
                    validations_append(sym, 'symbol_size_error')

    def find_wave_color(self, debug=False):
        if self.vantage_data:
            vantage_color = self.vantage_data["Wave_Colour"]

            self.process_pdf.wave_color_vantage = vantage_color # Store the color for potential future pages with no vantage data
        elif self.process_pdf.wave_color_vantage:
            vantage_color = self.process_pdf.wave_color_vantage # Get the color from a previous vantage file match
        else:
            vantage_color = None
        if vantage_color == 'Not Applicable':
            vantage_color = None
        try:
            # Extract the number inside parentheses from Wave_Colour
            vantage_pantone = vantage_color.split('(')[1].split(')')[0] + ' C'
            # get the RGB value from the pantone color
            vantage_rgb = pantone_to_rgb[vantage_pantone]
        except (KeyError, AttributeError): # When color is None, "Not Applicable" or does not map to a valid pantone
            vantage_rgb = None

        if vantage_rgb:
            t0=time()
            most_common_color = get_common_colors(self.images['no_text'])
            print("common color detection time: ", round(time()-t0, 2))
            
            if most_common_color:
                detected_rgb = most_common_color[::-1]

                dist = distance.euclidean(detected_rgb, vantage_rgb)

                # Convert RGB colors to LAB
                detected_lab = cv2.cvtColor(np.uint8([[detected_rgb]]), cv2.COLOR_RGB2LAB)[0][0]
                vantage_lab = cv2.cvtColor(np.uint8([[vantage_rgb]]), cv2.COLOR_RGB2LAB)[0][0]
                lab_dist = distance.euclidean(detected_lab, vantage_lab)

                # Convert RGB colors to HSV
                detected_hsv = cv2.cvtColor(np.uint8([[detected_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
                vantage_hsv = cv2.cvtColor(np.uint8([[vantage_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
                hsv_dist = distance.euclidean(detected_hsv, vantage_hsv)
                hue_dist = circular_distance(detected_hsv[0], vantage_hsv[0])

                if debug:
                    print(most_common_color)
                    print(f"Wave color (vantage RGB): {vantage_rgb}")
                    print(f"Wave color (detected RGB): {detected_rgb}")
                    print(f"RGB Distance: {dist}")
                    print(f"LAB Distance: {lab_dist}")
                    print(f"HSV Distance: {hsv_dist}")
                    print(f"Hue Distance: {hue_dist}")

                self.wave_color = {'detected' : detected_rgb,
                                'vantage' : vantage_rgb,
                                'vantage_pantone' : vantage_color,
                                'match' : dist<15}
            else:
                print("No wave found")
                self.wave_color = {'detected' : None,
                                'vantage' : vantage_rgb,
                                'vantage_pantone' : vantage_color,
                                'match' : None}
        else: #No vantage data and no previous match
            self.wave_color = {'detected' : None,
                                'vantage' : None,
                                'vantage_pantone' : None,
                                'match' : None}
    def draw_annotations(self):
        """
        Draw bounding boxes around all sub_images, text_data, and symbol_data.
        """
        annotated_image = self.images['annotated']
        # annotated_image = self.images['no_text']

        # Draw bounding boxes for sub_images
        # for sub_image in self.sub_images:
        #     bbox = sub_image["bbox"]
        #     color = (0, 255, 0) if sub_image["isBarcode"] else (255, 0, 0)
        #     cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # Draw bounding boxes for text_data
        # for text in self.text_data:
        #     bbox = text["bbox"]
        #     cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        
        # Draw bounding boxes for barcodes
        for barcode in self.barcodes:
            bbox = barcode["bbox"]
            cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 2)

        # Draw bounding boxes for symbol_data
        # for symbol in self.symbol_data:
        #     bbox = symbol["bbox"]
        #     if symbol["symbol_type"]=="Unknown":
        #         color=(255, 255, 0)
        #     elif symbol["symbol_type"]=="Image":
        #         color=(0, 0, 160)
        #         # color=(255, 0, 255)
        #     else:
        #         color=(255, 0, 0)
            
        #     cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # draw OD symbols
        for symbol in self.symbol_data:
            bbox = symbol["bbox"]
            # if True:
            # if symbol["symbol_name"].startswith('NP') or symbol["symbol_name"].startswith('PLED'):
            symbol_name = symbol["symbol_name"]
            if symbol["symbol_type"] == 'Image':
                color=(255, 0, 255)
                confidence = symbol["confidence"]
                cv2.putText(annotated_image, str(round(confidence, 2)), (bbox[2], bbox[3]), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 1, cv2.LINE_AA)
            else:
                color=(255,0,0)
                # np_name = symbol["symbol_name"].split('_')[1]
            cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color)
            cv2.putText(annotated_image, symbol_name, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        # Draw needles
        for needle in self.needle_data:
            bbox = needle["bbox"]
            color=(255, 0, 255)
            confidence = needle["confidence"]
            cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color)
            cv2.putText(annotated_image, str(round(confidence, 2)), (bbox[2], bbox[3]), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 1, cv2.LINE_AA)

        # Draw annotations (markup/redline)
        if self.annotations:
            for annot in self.annotations:
                bbox = annot["bbox"]
                color=(0, 0, 255)
                cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color)
                cv2.putText(annotated_image, annot['type'], (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)


        # # Remove needle point (fill with white)
        # for symbol in self.symbol_data:
        #     bbox = symbol["bbox"]
        #     if symbol["symbol_name"].startswith('NP'):
        #         color=(255, 255, 255)
        #         cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, -1)

        self.images['annotated'] = annotated_image

    def save_images(self, output_folder, page_num):
        """
        Save layer images per page
        For testing
        """

        self.draw_annotations()

        output_folder = os.path.join(output_folder, "page_images")
        if not os.path.exists(output_folder) and self.process_pdf.mode=='local':
            os.makedirs(output_folder)
        for key, image in self.images.items():
            if self.process_pdf.mode =='local' and key in ['changes', 'original', 'no_text', 'annotated', 'annotations', 'diff_image_original', 'diff_image_no_text']:
                cv2.imwrite(os.path.join(output_folder, f"{page_num}_{self.page_type}_{key}.png"), image)
            elif self.process_pdf.mode == 'blob' and key in ['changes', 'original', 'annotations']:
                write_img(os.path.join(output_folder, f"{page_num}_{self.page_type}_{key}.png"), image)


class ComparePages():
    def __init__(self, from_page, to_page):
        self.from_page = from_page
        self.to_page = to_page
        self.pixels_to_cm = to_page.pixels_to_cm
        
        self.ctq_moved_thresh = round(.1 / self.pixels_to_cm) # Threshold for CTQ items. Number in mm
        self.moved_thresh = round(.5 / self.pixels_to_cm) # Threshold for all other items. Number in mm

        # Auto-alignment between the from and to pages.
        # By default both pages are aligned by their top-left corner (pixmap
        # origin). auto_align() replaces this with an image-based alignment so a
        # global page misregistration (or different page sizes) does not inflate
        # the measured movement of every element.
        self.align_scale = 1.0      # Down-scale passed to auto_align_pages (tune to evaluate speed/accuracy)
        self.alignment_vector = (0, 0)  # (dx, dy) added to every 'to' coordinate

    def auto_align(self):
        """
        Compute the global alignment vector (to -> from) via template matching so
        that element movement is measured between truly corresponding positions
        instead of relying on the (sometimes imperfect) top-left page corner.
        """
        # self.alignment_vector = auto_align_pages(self.from_page.images.get('original'),
        #                                          self.to_page.images.get('original'),
        #                                          scale=self.align_scale)
        # print(f"Page alignment vector (to->from): {self.alignment_vector}")

    def compare(self):
        #self.compare_dimensions() # Disabled as per business requirement
        self.auto_align()
        self.process_pdf.timer.append("auto_align")
        self.compare_images('original')
        self.process_pdf.timer.append("compare_images")
        self.compare_images('no_text')
        self.process_pdf.timer.append("compare_images")
        self.compare_needles()
        self.process_pdf.timer.append("compare_needles")
        self.compare_symbols()
        self.process_pdf.timer.append("compare_symbols")
        self.compare_barcodes()
        self.process_pdf.timer.append("compare_barcodes")
        self.compare_generic_text()
        text_comparison = self.compare_input_text_elements()
        self.text_comparison.extend(text_comparison)
        self.process_pdf.timer.append("compare_input_text_elements")
        self.compare_translations()
        self.process_pdf.timer.append("compare_translations")
        self.compare_addresses()
        self.process_pdf.timer.append("compare_addresses")
        self.compare_wave_color()
        self.process_pdf.timer.append("compare_wave_color")
        self.collect_response()
        self.process_pdf.timer.append("collect_response")
        self.draw_changes()
        self.process_pdf.timer.append("draw_changes")
        pass # For debugging breakpoints

    def compare_dimensions(self):
        # Check that pages are the same dimensions
        from_width = self.from_page.page.rect.width
        from_height = self.from_page.page.rect.height
        to_width = self.to_page.page.rect.width
        to_height = self.to_page.page.rect.height

        dim_margin = .05
        if to_width < from_width * (1-dim_margin) or to_width > from_width * (1+dim_margin) or \
            to_height < from_height * (1-dim_margin) or to_height > from_height * (1+dim_margin):
            raise PageError(f"Page dimensions do not match. From: {from_width}x{from_height}, To: {to_width}x{to_height}")

    def compare_images(self, source):
        """
        Compare from and to images pixel by pixel.
        The 'to' image is first translated by the alignment vector so both pages
        are registered in the same (from) frame, instead of relying on the
        top-left corner. This prevents a global page offset from being flagged as
        added/deleted content along every edge.
        - Red = added elements
        - Blue = deleted elements
        - Green = color changes
        """
        from_image = self.from_page.images[source]
        to_image = self.to_page.images[source]

        # Ensure same size
        if from_image.shape != to_image.shape:
            return

        # Shift the 'to' image into the 'from' frame using the alignment vector.
        # The translation is integer so a nearest-neighbour warp is exact, and
        # exposed borders are filled with white so they count as background.
        dx, dy = self.alignment_vector
        if dx or dy:
            h, w = to_image.shape[:2]
            translation = np.float32([[1, 0, dx], [0, 1, dy]])
            to_image = cv2.warpAffine(to_image, translation, (w, h),
                                      flags=cv2.INTER_NEAREST,
                                      borderValue=(255, 255, 255))

        # Convert to int16 to avoid overflow in subtraction
        from_image = from_image.astype(np.int16)
        to_image = to_image.astype(np.int16)

        # White threshold: consider anything close to pure white as background
        white_thresh = 240
        mask1 = np.all(from_image > white_thresh, axis=-1)  # background in from_image
        mask2 = np.all(to_image > white_thresh, axis=-1)  # background in to_image

        # Added = background in from_image, but not in to_image
        added_mask = mask1 & (~mask2)

        # Deleted = background in to_image, but not in from_image
        deleted_mask = mask2 & (~mask1)

        # Color change = both non-background but color difference > threshold
        color_thresh = 50
        #diff = np.linalg.norm(from_image - to_image, axis=-1) # Euclidean distance in RGB space
        # Squared difference (int32, avoids float64 blowup)
        diff_sq = np.sum((from_image - to_image) ** 2, axis=-1, dtype=np.int32)
        # Compare squared distance to squared threshold (no sqrt needed)
        color_thresh_sq = color_thresh * color_thresh
        color_change_mask = (~mask1) & (~mask2) & (diff_sq > color_thresh_sq)

        # Output image initialized to white
        diff_image = np.ones_like(from_image, dtype=np.uint8) * 255

        # Mark changes
        diff_image[added_mask] = (0, 0, 255)    # Red
        diff_image[deleted_mask] = (255, 0, 0)  # Blue
        diff_image[color_change_mask] = (0, 255, 0)  # Green

        self.to_page.images['diff_image_'+source] = diff_image

    def compare_needles(self):
        """
        Compare needle positions between from_page and to_page using their centers and images
        """
        def make_comp_dict_needles(from_needle, to_needle, status):   
            if from_needle and to_needle:
                # Get the cropped images at their original sizes
                template = from_needle['crop']
                target = to_needle['crop']
                
                # Calculate size difference percentage
                size_diff = abs(1.0 - (template.shape[0] * template.shape[1]) / (target.shape[0] * target.shape[1]))
                
                # Try different scales for template matching
                scales = [0.95, 1.0, 1.05]
                best_score = -1
                score_thresh = .5
                
                for scale in scales:
                    scaled_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
                    if scaled_template.shape[0]>target.shape[0] and scaled_template.shape[1]>target.shape[1]:
                        result = cv2.matchTemplate(scaled_template, target, cv2.TM_CCOEFF_NORMED)
                        score = result.max()
                        if score > best_score:
                            best_score = score
                            if best_score > score_thresh:
                                break
                    elif scaled_template.shape[0]<=target.shape[0] and scaled_template.shape[1]<=target.shape[1]:
                        result = cv2.matchTemplate(target, scaled_template, cv2.TM_CCOEFF_NORMED)
                        score = result.max()
                        if score > best_score:
                            best_score = score
                            if best_score > score_thresh:
                                break
                    else: # Skip if can't fit template in target or viceversa
                        pass
                # If sizes differ by more than 20%, consider them different needles
                if best_score > score_thresh and size_diff < .2:
                    validations = 'Pass'
                else:
                    validations = 'needle_match_error'
                from_center = from_needle['center']
                to_center = align_center(to_needle['center'], self.alignment_vector)
                dist_cm = distance.euclidean(from_center, to_center)*self.pixels_to_cm
                dist_pct = percent_distance(from_needle['bbox'], align_bbox(to_needle['bbox'], self.alignment_vector))
            else:
                validations = None
                dist_cm = None
                dist_pct = None

            dict_set_none(from_needle, 'status', status)
            dict_set_none(to_needle, 'status', status)
            dict_set_none(from_needle, 'distance_pct', dist_pct)
            dict_set_none(to_needle, 'distance_pct', dist_pct)

            comp_dict = {
                    'from_bbox': dict_get_none(from_needle, 'bbox'),
                    'to_bbox': dict_get_none(to_needle, 'bbox'),
                    'validations' : validations,
                    'status': status,
                    'distance_cm': dist_cm,
                    'distance_pct': dist_pct
                }
            return comp_dict
        
        # Extract center points of needle bounding boxes
        from_centers = [needle['center'] for needle in self.from_page.needle_data]
        to_centers = [align_center(needle['center'], self.alignment_vector) for needle in self.to_page.needle_data]

        # Match needle centers between pages
        matches, removed, added, moved = match_points(from_centers, to_centers, moved_thresh=self.moved_thresh,
                                                      method = "translation_discrete")

        changes = {"Added": 0, "Deleted": 0, "Moved": 0, "Same": 0}
        needle_comparison = []

        # Mark non-matches as either Added or Deleted
        for del_idx in removed:
            status = 'Deleted'
            changes[status] += 1
           
            comp_dict = make_comp_dict_needles(self.from_page.needle_data[del_idx], None, status)
            needle_comparison.append(comp_dict)

        for add_idx in added:
            status = 'Added'
            changes[status] += 1
        
            comp_dict = make_comp_dict_needles(None, self.to_page.needle_data[add_idx], status)
            needle_comparison.append(comp_dict)

        # Compare matched needle images using template matching
        for i, match in enumerate(matches):
            from_idx = match[0]
            to_idx = match[1]
            status = 'Same' if not moved[i] else 'Moved'

            comp_dict = make_comp_dict_needles(self.from_page.needle_data[from_idx], self.to_page.needle_data[to_idx], status)
            needle_comparison.append(comp_dict)

        self.needle_comparison = needle_comparison
        print(f"Needle changes: {changes}")
        return changes

    def compare_symbols(self):
        """
        Compare symbols between from_page and to_page using match_points()
        """
        def make_comp_dict_symbols(from_symbol, to_symbol, status):  
            validations = [] # Comparison validations (from vs to)
            validations_to = validations_join(to_symbol, None) # 'to' symbol data-accuracy validations (e.g. ACCLM)
            if from_symbol and to_symbol:
                from_center = from_symbol['center']
                to_center = align_center(to_symbol['center'], self.alignment_vector)
                dist_cm = distance.euclidean(from_center, to_center)*self.pixels_to_cm
                dist_pct = percent_distance(from_symbol['bbox'], align_bbox(to_symbol['bbox'], self.alignment_vector))
                
                # if to_symbol['font'] != from_symbol['font'] and to_symbol['font']!="Image_Symbol" and from_symbol['font']!="Image_Symbol":
                #     validations.append("fontstyle_error")
                # if to_symbol['size'] != from_symbol['size'] and to_symbol['font']!="Image_Symbol" and from_symbol['font']!="Image_Symbol":
                #     validations.append("fontsize_error")
                if not same_color(to_symbol['color'], from_symbol['color']):
                    validations.append("symbolcolor_change")

            else:
                dist_pct = None
                dist_cm = None
            
            dict_set_none(from_symbol, 'distance_pct', dist_pct)
            dict_set_none(to_symbol, 'distance_pct', dist_pct)
            dict_set_none(from_symbol, 'status', status)
            dict_set_none(to_symbol, 'status', status)

            symbol_name = dict_get_none(from_symbol, 'symbol_name')
            if not symbol_name:
                symbol_name = dict_get_none(to_symbol, 'symbol_name')

            # Comparison dictionary
            comp_dict = {'symbol_name' : get_symbol_business_name(symbol_name), # Use business name if available
                            'from_bbox' : dict_get_none(from_symbol,'bbox'),
                            'to_bbox' : dict_get_none(to_symbol,'bbox'),
                            'from_font' : dict_get_none(from_symbol,'font'),
                            'to_font' : dict_get_none(to_symbol,'font'),
                            'from_size': dict_get_none(from_symbol,'size'),
                            'to_size': dict_get_none(to_symbol,'size'),
                            'validations' : validations_str(validations) if from_symbol and to_symbol else None,
                            'validations_to' : value_accuracy_str(validations_to) if to_symbol else None,
                            'status' : status,
                            'distance_cm': dist_cm,
                            'distance_pct': dist_pct
                        }
            return comp_dict
        
        # Extract centers and group by symbol name
        symbol_types = set(s["symbol_name"] for s in self.from_page.symbol_data + self.to_page.symbol_data)
        changes = {"Added": 0, "Deleted": 0, "Moved": 0, "Same": 0}
        symbol_comparison = []
            
        # Compare symbols of each type separately
        for symbol_type in symbol_types:
            from_symbols = [s for s in self.from_page.symbol_data if s["symbol_name"] == symbol_type]
            to_symbols = [s for s in self.to_page.symbol_data if s["symbol_name"] == symbol_type]
            
            from_centers = [s["center"] for s in from_symbols]
            to_centers = [align_center(s["center"], self.alignment_vector) for s in to_symbols]
            
            if symbol_type in ['REF', 'LOT', 'EXPIRY']:
                moved_thresh = self.ctq_moved_thresh
            else:
                moved_thresh = self.moved_thresh

            # Match symbols using their centers
            matches, removed, added, moved = match_points(from_centers, to_centers, moved_thresh=moved_thresh)
            
            # Update statuses based on matching results
            for del_idx in removed:
                status = "Deleted"
                changes["Deleted"] += 1
                comp_dict = make_comp_dict_symbols(from_symbols[del_idx], None, status)
                symbol_comparison.append(comp_dict)
                
            for add_idx in added:
                status = 'Added'
                changes["Added"] += 1
                comp_dict = make_comp_dict_symbols(None, to_symbols[add_idx], status)
                symbol_comparison.append(comp_dict)

            # Update matched pairs
            for i, match in enumerate(matches):
                from_idx = match[0]
                to_idx = match[1]

                status = 'Same' if not moved[i] else 'Moved'
                changes[status] += 1

                comp_dict = make_comp_dict_symbols(from_symbols[from_idx], to_symbols[to_idx], status)
                symbol_comparison.append(comp_dict)

        self.symbol_comparison = symbol_comparison
        print(f"Symbol changes: {changes}")
        return changes

    def compare_barcodes(self):
        """
        Compare barcodes between from_page and to_page using match_points()
        """
        def make_comp_dict_barcodes(from_barcode, to_barcode, status): 
            # Get previous validations (only TO barcode)
            validations_to = validations_join(to_barcode, None)
            validations = []
            if from_barcode and to_barcode:    
                from_center = bbox_center(from_barcode['bbox'])
                to_center = align_center(bbox_center(to_barcode['bbox']), self.alignment_vector)
                dist_cm = distance.euclidean(from_center, to_center)*self.pixels_to_cm
                dist_pct = percent_distance(from_barcode['bbox'], align_bbox(to_barcode['bbox'], self.alignment_vector))
                
                dict_set_none(from_barcode, 'distance_pct', dist_pct)
                dict_set_none(to_barcode, 'distance_pct', dist_pct)
            
                if to_barcode['decoded'] and from_barcode['decoded']:
                    if len(to_barcode['decoded']) != len(from_barcode['decoded']):
                        validations.append('length_error')
            else:
                dist_pct = None
                dist_cm = None
            
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
                'validations': validations_str(validations) if from_barcode and to_barcode else None,
                'validations_to': value_accuracy_str(validations_to, expected_value=expected_value) if to_barcode else None,
                'grade' : dict_get_none(to_barcode, 'grade'),
                'grade_metrics' : dict_get_none(to_barcode, 'grade_metrics'),
                'subtype' : dict_get_none(to_barcode, 'subtype'),
                'crop_filename' : dict_get_none(to_barcode, 'crop_filename'),
                'status': status,
                'distance_cm': dist_cm,
                'distance_pct': dist_pct
            }
            return comp_dict
        # Extract barcode types from both pages
        barcode_types = set(b["type"] for b in self.from_page.barcodes + self.to_page.barcodes)
        changes = {"Added": 0, "Deleted": 0, "Moved": 0, "Same": 0}
        barcode_comparison = []
        
        # Compare barcodes of each type separately
        for barcode_type in barcode_types:
            from_barcodes = [b for b in self.from_page.barcodes if b["type"] == barcode_type]
            to_barcodes = [b for b in self.to_page.barcodes if b["type"] == barcode_type]
            
            from_centers = [bbox_center(b["bbox"]) for b in from_barcodes]
            to_centers = [align_center(bbox_center(b["bbox"]), self.alignment_vector) for b in to_barcodes]
            
            # Match barcodes using their centers
            matches, removed, added, moved = match_points(from_centers, to_centers, moved_thresh=self.ctq_moved_thresh,
                                                          method="translation_discrete",
                                                          dist_thresh=round(4/self.pixels_to_cm))
            
            # Handle removed barcodes
            for del_idx in removed:
                status = "Deleted"
                changes["Deleted"] += 1
                comp_dict = make_comp_dict_barcodes(from_barcodes[del_idx], None, status)
                barcode_comparison.append(comp_dict)
            
            # Handle added barcodes
            for add_idx in added:
                status = "Added"
                changes["Added"] += 1
                comp_dict = make_comp_dict_barcodes(None, to_barcodes[add_idx], status)
                barcode_comparison.append(comp_dict)

            # Handle matched pairs
            for i, match in enumerate(matches):
                from_idx = match[0]
                to_idx = match[1]

                status = 'Same' if not moved[i] else 'Moved'
                changes[status] += 1

                comp_dict = make_comp_dict_barcodes(from_barcodes[from_idx], to_barcodes[to_idx], status)
                barcode_comparison.append(comp_dict)

        self.barcode_comparison = barcode_comparison
        print(f"Barcode changes: {changes}")
        return changes

    def compare_input_text_elements(self):
        """
        Compare text elements between from_page and to_page based on TAD matches.
        Only compares text elements (not symbols).
        """
        def make_comp_dict_text_elements(from_text_element, to_text_element, status, col_name, tad_value, element): 
            # Get previous validations (if any). Join duplicates
            validations_to = validations_join(to_text_element, None)   
            validations = []
            if from_text_element and to_text_element:
                from_center = from_text_element['center']
                to_center = align_center(to_text_element['center'], self.alignment_vector)
                dist_cm = distance.euclidean(from_center, to_center)*self.pixels_to_cm
                dist_pct = percent_distance(from_text_element['bbox'], align_bbox(to_text_element['bbox'], self.alignment_vector))
                                
                if to_text_element['text'].strip() != from_text_element['text'].strip():
                    validations.append("value_change")
                validations, validations_to = font_check(validations, validations_to, from_text_element, to_text_element, self.process_pdf.document_type)
            else:
                dist_pct = None
                dist_cm = None
                if to_text_element:  # 'Added' element: still run font_check so the size-threshold check applies
                    validations, validations_to = font_check(validations, validations_to, from_text_element, to_text_element, self.process_pdf.document_type)
            
            dict_set_none(from_text_element, 'status', status)
            dict_set_none(to_text_element, 'status', status)
            dict_set_none(from_text_element, 'distance_pct', dist_pct)
            dict_set_none(to_text_element, 'distance_pct', dist_pct)

            comp_dict = {
                'label_element': col_name,
                'input_value' : tad_value,
                'from_text': dict_get_none(from_text_element, 'text'),
                'to_text': dict_get_none(to_text_element, 'text'),
                'from_bbox': dict_get_none(from_text_element, 'bbox'),
                'to_bbox': dict_get_none(to_text_element, 'bbox'),
                'from_font': dict_get_none(from_text_element, 'font'),
                'to_font': dict_get_none(to_text_element, 'font'),
                'from_size': dict_get_none(from_text_element, 'size'),
                'to_size': dict_get_none(to_text_element, 'size'),
                'validations' : validations_str(validations) if from_text_element and to_text_element else None,
                'validations_to': value_accuracy_str(validations_to, tad_value) if to_text_element else None,
                'status': status,
                'distance_cm': dist_cm,
                'distance_pct': dist_pct
            }
            return comp_dict
        # List of text-only TAD elements to compare
        text_elements = [
            'brand', 'len_cm_to', 'len_in_to', 'USP_to', 'gauge_to',
            'needle_type_to', 'needle_len_lbl_to', #'needle_len_l1_to',
            'curve_frac', 'needle_point_tm', 'needle_tm', 'needle_alloy_to', 'pieces_to',
            'num_strands_to',
            'REF_no', 
            'LOT_no', 'MAN_date', 'EXP_date',
            'CCN', 
            # Vantage elements
            'Raw_Material', 
            # Generic elements
            #'generic'
            #'ethicon',
        ]

        changes = {"Added": 0, "Deleted": 0, "Moved": 0, "Same": 0}

        text_comparison = []

        for element in text_elements:
            # if element == "generic":
            #     continue
            # Compare text elements for each TAD field
            from_matches = self.from_page.tad_matches.get(element, [])
            to_matches = self.to_page.tad_matches.get(element, [])

            from_centers = [match["center"] for match in from_matches]
            to_centers = [align_center(match["center"], self.alignment_vector) for match in to_matches]

            # Get original value from TAD, Vantage or ps5 data
            tad_value, col_name = self.get_data_value(element)

            # Match elements using their centers
            matches, removed, added, moved = match_points(from_centers, to_centers, moved_thresh=self.moved_thresh)

            # Handle removed elements
            for del_idx in removed:
                status = 'Deleted'
                changes["Deleted"] += 1

                comp_dict = make_comp_dict_text_elements(from_matches[del_idx], None, status, col_name, tad_value, element)
                text_comparison.append(comp_dict)

            # Handle added elements
            for add_idx in added:
                status = 'Added'
                changes["Added"] += 1
                
                comp_dict = make_comp_dict_text_elements(None, to_matches[add_idx], status, col_name, tad_value, element)
                text_comparison.append(comp_dict)

            # Handle matched pairs
            for i, match in enumerate(matches):
                from_idx = match[0]
                to_idx = match[1]
            
                status = "Same" if not moved[i] else "Moved"
                changes[status] += 1

                comp_dict = make_comp_dict_text_elements(from_matches[from_idx], to_matches[to_idx], status, col_name, tad_value, element)
                text_comparison.append(comp_dict)

        self.text_elements = text_elements
        print(f"Text element changes: {changes}")
        return text_comparison

    def compare_generic_text(self):
        def make_comp_dict_text_elements(from_text_element, to_text_element, status, col_name, tad_value):
            # Get previous validations (if any). Join duplicates
            validations_to = validations_join(to_text_element, None) 
            validations = []
            if from_text_element and to_text_element:
                from_center = from_text_element['center']
                to_center = align_center(to_text_element['center'], self.alignment_vector)
                dist_cm = distance.euclidean(from_center, to_center)*self.pixels_to_cm
                dist_pct = percent_distance(from_text_element['bbox'], align_bbox(to_text_element['bbox'], self.alignment_vector))
                                
                validations, validations_to = font_check(validations, validations_to, from_text_element, to_text_element, self.process_pdf.document_type)

                if from_text_element['text'].strip() != to_text_element['text'].strip():
                    validations.append("value_change")
            else:
                dist_pct = None
                dist_cm = None
                if to_text_element:  # 'Added' element: still run font_check so the size-threshold check applies
                    validations, validations_to = font_check(validations, validations_to, from_text_element, to_text_element, self.process_pdf.document_type)
            
            dict_set_none(from_text_element, 'status', status)
            dict_set_none(to_text_element, 'status', status)
            dict_set_none(from_text_element, 'distance_pct', dist_pct)
            dict_set_none(to_text_element, 'distance_pct', dist_pct)

            comp_dict = {
                'label_element': col_name,
                'input_value' : tad_value,
                'from_text': dict_get_none(from_text_element, 'text'),
                'to_text': dict_get_none(to_text_element, 'text'),
                'from_bbox': dict_get_none(from_text_element, 'bbox'),
                'to_bbox': dict_get_none(to_text_element, 'bbox'),
                'from_font': dict_get_none(from_text_element, 'font'),
                'to_font': dict_get_none(to_text_element, 'font'),
                'from_size': dict_get_none(from_text_element, 'size'),
                'to_size': dict_get_none(to_text_element, 'size'),
                'validations' : validations_str(validations) if from_text_element and to_text_element else None,
                'validations_to': value_accuracy_str(validations_to) if to_text_element else None,
                'status': status,
                'distance_cm': dist_cm,
                'distance_pct': dist_pct
            }
            return comp_dict

        changes = {"Added": 0, "Deleted": 0, "Moved": 0, "Same": 0}

        text_comparison = []
        fuzzy_threshold = 80

        unique_texts_from = set([e['text'].strip().upper() for e in self.from_page.text_data] )
        unique_texts_to = set([e['text'].strip().upper() for e in self.to_page.text_data] )
        for text in unique_texts_from.union(unique_texts_to):
            # from_matches = [e for e in self.from_page.text_data if fuzz.ratio(e['text'].strip().upper(), text) >= fuzzy_threshold]
            from_matches = []
            for e in self.from_page.text_data[::-1]:
                normalized_text = e['text'].strip().upper()
                # Exact match or fuzzy match provided that there is no exact match possible
                if normalized_text == text or \
                    fuzz.ratio(normalized_text, text) >= fuzzy_threshold and normalized_text not in unique_texts_to:
                    from_matches.append(e)
                    self.from_page.text_data.remove(e)
            # to_matches = [e for e in self.to_page.text_data if fuzz.ratio(e['text'].strip().upper(), text) >= fuzzy_threshold]
            to_matches = []
            for e in self.to_page.text_data[::-1]:
                normalized_text = e['text'].strip().upper()
                if normalized_text == text or \
                    fuzz.ratio(normalized_text, text) >= fuzzy_threshold and normalized_text not in unique_texts_from:
                    to_matches.append(e)
                    self.to_page.text_data.remove(e)

            from_centers = [match["center"] for match in from_matches]
            to_centers = [align_center(match["center"], self.alignment_vector) for match in to_matches]

            # Get original value from TAD, Vantage or ps5 data
            tad_value, col_name = '', 'generic'

            # Match elements using their centers
            matches, removed, added, moved = match_points(from_centers, to_centers, moved_thresh=self.moved_thresh)

            # Handle removed elements
            for del_idx in removed:
                status = 'Deleted'
                changes["Deleted"] += 1

                comp_dict = make_comp_dict_text_elements(from_matches[del_idx], None, status, col_name, tad_value)
                text_comparison.append(comp_dict)

            # Handle added elements
            for add_idx in added:
                status = 'Added'
                changes["Added"] += 1
                
                comp_dict = make_comp_dict_text_elements(None, to_matches[add_idx], status, col_name, tad_value)
                text_comparison.append(comp_dict)

            # Handle matched pairs
            for i, match in enumerate(matches):
                from_idx = match[0]
                to_idx = match[1]
            
                status = "Same" if not moved[i] else "Moved"
                changes[status] += 1

                comp_dict = make_comp_dict_text_elements(from_matches[from_idx], to_matches[to_idx], status, col_name, tad_value)
                text_comparison.append(comp_dict)

        self.text_comparison = text_comparison
        print(f"Text element changes: {changes}")
        return changes

    def compare_translations(self):
        """
        Compare translation blocks between from_page and to_page using match_points()
        """
        def make_comp_dict_translations(from_trans, to_trans, status):   
            if from_trans and to_trans:            
                # Calculate distance for matched pairs
                from_center = bbox_center(from_trans['bbox'])
                to_center = align_center(bbox_center(to_trans['bbox']), self.alignment_vector)
                dist_cm = distance.euclidean(from_center, to_center) * self.pixels_to_cm
                dist_pct = percent_distance(from_trans['bbox'], align_bbox(to_trans['bbox'], self.alignment_vector))
            else:
                dist_cm = None
                dist_pct = None
            
            dict_set_none(from_trans, 'distance_pct', dist_pct)
            dict_set_none(to_trans, 'distance_pct', dist_pct)
            dict_set_none(from_trans, 'status', status)
            dict_set_none(to_trans, 'status', status)
            if to_trans:
                validations_to = to_trans['validations']
            else:
                validations_to = None

            trans_type = dict_get_none(from_trans, 'type')
            trans_name = dict_get_none(from_trans, 'name')
            if not trans_type:
                trans_type = dict_get_none(to_trans, 'type')
                trans_name = dict_get_none(to_trans, 'name')

            comp_dict = {
                'type': trans_type,
                'name': trans_name,
                'from_bbox': dict_get_none(from_trans, 'bbox'),
                'to_bbox': dict_get_none(to_trans, 'bbox'),
                'validations_to': value_accuracy_str(validations_to) if to_trans else None,
                'status': status,
                'distance_cm': dist_cm,
                'distance_pct': dist_pct
            }  
            return comp_dict    
        # Extract and group by translation name (multipass, TP, etc)
        translation_types = set(t["name"] for t in self.from_page.translation_data + self.to_page.translation_data)
        changes = {"Added": 0, "Deleted": 0, "Moved": 0, "Same": 0}
        translation_comparison = []
        
        # Compare translations of each type separately
        for trans_type in translation_types:
            from_translations = [t for t in self.from_page.translation_data if t["name"] == trans_type]
            to_translations = [t for t in self.to_page.translation_data if t["name"] == trans_type]
            
            # Get centers of bounding boxes
            from_centers = [bbox_center(t["bbox"]) for t in from_translations]
            to_centers = [align_center(bbox_center(t["bbox"]), self.alignment_vector) for t in to_translations]
            
            # Match translations using their centers
            matches, removed, added, moved = match_points(from_centers, to_centers, moved_thresh=self.moved_thresh)
                        
            # Handle removed translations
            for del_idx in removed:
                status = "Deleted"
                changes[status] += 1
            
                comp_dict = make_comp_dict_translations(from_translations[del_idx], None, status)
                translation_comparison.append(comp_dict)
            
            # Handle added translations
            for add_idx in added:
                status = "Added"
                changes[status] += 1

                comp_dict = make_comp_dict_translations(None, to_translations[add_idx], status)
                translation_comparison.append(comp_dict)

            # Handle matched pairs
            for i, match in enumerate(matches):
                from_idx = match[0]
                to_idx = match[1]
            
                status = "Same" if not moved[i] else "Moved"
                changes[status] += 1

                comp_dict = make_comp_dict_translations(from_translations[from_idx], to_translations[to_idx], status)
                translation_comparison.append(comp_dict)

        self.translation_comparison = translation_comparison
        print(f"Translation changes: {changes}")
        return changes

    def compare_addresses(self):
        """
        Compare manufacturer address blocks between from_page and to_page using match_points()
        """
        def make_comp_dict_addresses(from_add, to_add, status):   
            # Get validations (Only TO address)
            validations = validations_join(to_add, None)
            if from_add and to_add:            
                # Calculate distance for matched pairs
                from_center = bbox_center(from_add['bbox'])
                to_center = align_center(bbox_center(to_add['bbox']), self.alignment_vector)
                dist_cm = distance.euclidean(from_center, to_center) * self.pixels_to_cm
                dist_pct = percent_distance(from_add['bbox'], align_bbox(to_add['bbox'], self.alignment_vector))
            else:
                dist_cm = None
                dist_pct = None
            
            dict_set_none(from_add, 'distance_pct', dist_pct)
            dict_set_none(to_add, 'distance_pct', dist_pct)
            dict_set_none(from_add, 'status', status)
            dict_set_none(to_add, 'status', status)

            add_name = dict_get_none(from_add, 'name')
            if not add_name:
                add_name = dict_get_none(to_add, 'name')

            comp_dict = {
                'name': add_name,
                'from_bbox': dict_get_none(from_add, 'bbox'),
                'to_bbox': dict_get_none(to_add, 'bbox'),
                'validations_to': value_accuracy_str(validations) if to_add else None,
                'status': status,
                'distance_cm': dist_cm,
                'distance_pct': dist_pct
            }  
            return comp_dict    
        # Extract and group by address name (Ethicon LLC Guaynabo, etc)
        address_names = set(t["name"] for t in self.from_page.address_data + self.to_page.address_data)
        changes = {"Added": 0, "Deleted": 0, "Moved": 0, "Same": 0}
        address_comparison = []
        
        # Compare addresses of each type separately
        for add_name in address_names:
            from_addresses = [t for t in self.from_page.address_data if t["name"] == add_name]
            to_addresses = [t for t in self.to_page.address_data if t["name"] == add_name]
            
            # Get centers of bounding boxes
            from_centers = [bbox_center(t["bbox"]) for t in from_addresses]
            to_centers = [align_center(bbox_center(t["bbox"]), self.alignment_vector) for t in to_addresses]
            
            # Match addresses using their centers
            matches, removed, added, moved = match_points(from_centers, to_centers, moved_thresh=self.moved_thresh)
                        
            # Handle removed addresses
            for del_idx in removed:
                status = "Deleted"
                changes[status] += 1
            
                comp_dict = make_comp_dict_addresses(from_addresses[del_idx], None, status)
                address_comparison.append(comp_dict)
            
            # Handle added translations
            for add_idx in added:
                status = "Added"
                changes[status] += 1

                comp_dict = make_comp_dict_addresses(None, to_addresses[add_idx], status)
                address_comparison.append(comp_dict)

            # Handle matched pairs
            for i, match in enumerate(matches):
                from_idx = match[0]
                to_idx = match[1]
            
                status = "Same" if not moved[i] else "Moved"
                changes[status] += 1

                comp_dict = make_comp_dict_addresses(from_addresses[from_idx], to_addresses[to_idx], status)
                address_comparison.append(comp_dict)

        self.address_comparison = address_comparison
        print(f"Manufacturer address changes: {changes}")
        return changes 

    def compare_wave_color(self):
        if self.to_page.wave_color['match'] == None:
            validations = None
        elif self.to_page.wave_color['match']:
            validations = []
        else:
            validations = ['wave_color_error', 'thunderbird_data_error']

        wave_color_comparison = {'vantage_pantone' : self.to_page.wave_color['vantage_pantone'],
                                 'vantage_RGB' : self.to_page.wave_color['vantage'],
                                 'from_detected' : self.from_page.wave_color['detected'],
                                 'to_detected' : self.to_page.wave_color['detected'],
                                 'from_match' : self.from_page.wave_color['match'],
                                 'to_match' : self.to_page.wave_color['match'],
                                 'validations_to' : value_accuracy_str(validations, self.to_page.wave_color['vantage_pantone'])
                                 }
        if self.to_page.wave_color['vantage']==None:
            # Ignore wave_color results if no vantage color available (could be from color being "Not Applicable")
            self.wave_color_comparison = False
        else:
            self.wave_color_comparison = wave_color_comparison

    def get_data_value(self, key):
        try:  
            if self.to_page.tad_data: 
                tad_value = self.to_page.tad_data[key]
            else:
                tad_value="N/A"
            col_name = self.process_pdf.colname_map[key].replace("To ", "").replace("Future ", "")        
        except KeyError:
            pass
        # CCN special case
        if key=="CCN":
            tad_value="N/A"
            col_name = "CCN"
        # PS5 elements
        elif key in ['curve_frac']:
            if self.to_page.ps5_data: 
                tad_value = self.to_page.ps5_data[key] 
            else: 
                tad_value = None
            col_name = "Curve Fractional" # TO-DO fix
        elif key == 'needle_len_lbl_to' and not self.to_page.tad_data: # Case where tadless is True and no TAD data is available, but PS5 data is available
            if self.to_page.ps5_data: 
                tad_value = self.to_page.ps5_data['l1_lbl'] 
            else: 
                tad_value = None            
            col_name = self.process_pdf.colname_map[key].replace("To ", "").replace("Future ", "")
        # Vantage elements
        elif key in ['Raw_Material']:
            if self.to_page.vantage_data: 
                tad_value = self.to_page.vantage_data['Scientific Name']
            else:
                tad_value = None
            col_name = "Raw Material"
        elif key == 'LOT_no':
            tad_value = "N/A"
            col_name = "LOT number"
        elif key == 'MAN_date':
            tad_value = "N/A"
            col_name = "Manufacturing Date"
        elif key == 'EXP_date':
            tad_value = "N/A"
            col_name = "Expiry Date"
        return tad_value, col_name

    def collect_response(self):
        response = {}

        response["success"] = True
        response["error_message"] = ''

        response["CCN"] = {"from":self.from_page.ccn if self.from_page.ccn else 'Page'+str(self.to_page.page_num + 1),
                           "to":self.to_page.ccn if self.to_page.ccn else 'Page'+str(self.to_page.page_num + 1)}
        
        response["page_num"] = self.to_page.page_num + 1

        # Add extracted data from soudces
        response["ps5_data"] = self.to_page.ps5_data 
        response["vantage_data"] = self.to_page.vantage_data 
        # Filter and rename tad_data to use original Thunderbird column names
        if self.to_page.tad_data:
            filtered_tad_data = {}
            for internal_key, excel_col_name in self.process_pdf.colname_map.items():
                if internal_key in self.to_page.tad_data:
                    filtered_tad_data[excel_col_name] = self.to_page.tad_data[internal_key]
            response["tad_data"] = filtered_tad_data
        else:
            response["tad_data"] = None

        # Wave color
        if self.wave_color_comparison:
            response["wave_color"] = self.wave_color_comparison 

        # tad_results
        response["TAD_results"] = {"from":self.from_page.tad_results,
                                   "to":self.to_page.tad_results} if self.to_page.tad_data else None
        # Needle comparison
        response["needle_comparison"] = self.needle_comparison

        # Symbol comparison
        response["symbol_comparison"] = self.symbol_comparison

        # Barcode comparison
        response["barcode_comparison"] = self.barcode_comparison
        
        # Text comparison
        response["text_comparison"] = self.text_comparison

        # Translation comparison
        response["translation_comparison"] = self.translation_comparison

        # Address comparison
        response["address_comparison"] = self.address_comparison

        if self.from_page.annotations:
            response['annotations'] = self.from_page.annotations

        self.response = make_json_serializable(response)

    def draw_changes(self):
        """
        Draw changes in the labels based on the merged comparison data.
        """
        from_changes_image = self.from_page.images['original'].copy()
        to_changes_image = self.to_page.images['original'].copy()



        # Display distance moved to the left of the bounding box
        display_dist = False
        display_movement = False

        # Function to draw bounding boxes and distances
        def draw_element(image_from, image_to, element):
            colors = {
                "Same": (0, 255, 0),    # Green
                "Moved": (255, 0, 0),   # Blue
                "Deleted": (0, 0, 255), # Red
                "Added": (0, 0, 255),   # Red
                # "Changed": (255, 165, 0) # Orange
            }

            issue_color = (0, 165, 255)   
            color = colors[element['status']] if element.get('validations', None) in ['Pass', None] and (not element.get('validations_to', 'Match') or element.get('validations_to', 'Match').startswith('Match')) else issue_color

            bbox_from = element.get('from_bbox') 
            if bbox_from:
                cv2.rectangle(image_from, (bbox_from[0], bbox_from[1]), (bbox_from[2], bbox_from[3]), color, 2)
            bbox_to = element.get('to_bbox')
            if bbox_to:
                cv2.rectangle(image_to, (bbox_to[0], bbox_to[1]), (bbox_to[2], bbox_to[3]), color, 2)
            if display_dist and 'distance_pct' in element and bbox_to and bbox_from:
                image_to = draw_dist_pct(image_to, element, from_to="to")
                image_from = draw_dist_pct(image_from, element, from_to="from") 
            if display_movement and bbox_to and bbox_from:
                center_from = bbox_center(bbox_from)
                center_to = bbox_center(bbox_to)
                cv2.line(image_to, center_from, center_to, color, 2)
                cv2.line(image_from, center_from, center_to, color, 2)
                cv2.circle(image_from, center_from, 4, color, -1)
                cv2.circle(image_to, center_to, 4, color, -1)


        # Draw symbol comparisons
        if getattr(self, 'symbol_comparison', None):
            for symbol in self.symbol_comparison:
                draw_element(from_changes_image, to_changes_image, symbol)

        # Draw needle comparisons
        if getattr(self, 'needle_comparison', None):
            for needle in self.needle_comparison:
                draw_element(from_changes_image, to_changes_image, needle)

        # Draw barcode comparisons
        if getattr(self, 'barcode_comparison', None):
            for barcode in self.barcode_comparison:
                draw_element(from_changes_image, to_changes_image, barcode)

        # Draw text comparisons
        if getattr(self, 'text_comparison', None):
            for text in self.text_comparison:
                draw_element(from_changes_image, to_changes_image, text)

        # Draw translation comparisons
        if getattr(self, 'translation_comparison', None):
            for trans in self.translation_comparison:
                draw_element(from_changes_image, to_changes_image, trans)
        
        # Draw address comparisons
        if getattr(self, 'address_comparison', None):
            for address in self.address_comparison:
                draw_element(from_changes_image, to_changes_image, address)

        self.from_page.images['changes'] = from_changes_image
        self.to_page.images['changes'] = to_changes_image

