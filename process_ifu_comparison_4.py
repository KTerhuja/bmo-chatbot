import os
import sys
import json
import pandas as pd
from pathlib import Path
import numpy as np

# Setup Path
ROOT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(ROOT_DIR))
from utils_ifu.extraction import extract_data,get_pdf_type, get_legends_others_grid
from utils_ifu.load_models import load_llm, load_ocr
from utils_ifu.genai import generate_translations, translate_symbols
from utils_ifu.structural_analysis import create_difference_report, compare_texts, create_difference_report_ifu_comp
from utils_ifu.processing import restructure_text
from utils_ifu.semantic_analysis import perform_semantic_analysis
from utils_ifu.embeddings import match_lists_by_embeddings, match_lists, align_lists
from utils_ifu.symbol_legends import validate_symbols_and_detect_objects
from utils_ifu.legends_map import get_legends_map
from utils_ifu.utils import get_doc
from utils_ifu.load_models import load_yolo_model
from utils_ifu.ifu_comparison import *
from utils_ifu.symbol_name_mapping import acclm_mapping
from utils_ifu.debug_wordFIle import debug_word_file
from utils_ifu.ifu_grids import get_grid_bboxes
from utils_ifu.legends_extraction_grid import get_symbol_descriptors, get_symbol_legend_language_type
from utils_ifu.symbols_detection_grid import get_symbol_legend_bboxes
import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):   # covers int64, float64, bool_, etc.
            return obj.item()
        return super().default(obj)

class IFUComparison:
    def __init__(self):
        """Initialize the processor and persistent clients."""
        self.client_ocr = load_ocr() #
        self.llm = None
        self.llm_structured = None
        self.client_embed = None
        self.current_gpt_version = None
        self.response_json = {}
        yolo_path = os.path.join(ROOT_DIR, 'object_detection', 'best.pt')
        yolo_model = load_yolo_model(yolo_path)
        self.yolo_model = yolo_model

    def _initialize_llm(self, gpt_version):
        """Internal helper to load/switch LLM versions only when necessary."""
        if self.current_gpt_version != gpt_version:
            print(f"Loading {gpt_version}...")
            # Updated to pass the version to load_llm
            self.llm, self.llm_structured, self.client_embed = load_llm(gpt_version=gpt_version)
            self.current_gpt_version = gpt_version

    def process_file(self, pdf_path_1, pdf_path_2, gpt_version="gpt-5.1", mode="local", class_legend_mapping=None):
        """
        The main orchestration pipeline.
        gpt_version: Accepts "gpt-4.1" or "gpt-5.1"
        """
        
        p1 = pdf_path_1.upper()
        p2 = pdf_path_2.upper()

        if "CURRENT" in p1 and "FUTURE" in p2:
            file_path_current = pdf_path_1
            file_path_future = pdf_path_2
        elif "FUTURE" in p1 and "CURRENT" in p2:
            file_path_current = pdf_path_2
            file_path_future = pdf_path_1
        else:
            return {"File Name Error":"Please Make sure the file name has 'CURRENT' and 'FUTURE'"}

        # if "docx" in pdf_path_1:
        #     debug_word_file(pdf_path_1, "debug1.pdf")

        # if "docx" in pdf_path_2:
        
        #         debug_word_file(pdf_path_2, "debug2.pdf")

        pdf_type_current = get_pdf_type(file_path_current, mode=mode)
        pdf_type_future = get_pdf_type(file_path_future, mode=mode)

        # Ensure the correct LLM version is loaded
        self._initialize_llm(gpt_version)

        # 1. Data Extraction
        if pdf_type_current == "grid":
            data_json_current = self._extract_content(file_path_current, pdf_type_current, mode)
            data_json_current_clean = data_json_current.copy()
        elif pdf_type_current == "others":
            data_json_current = self._extract_content(file_path_current, pdf_type_current, mode)
            # data_json_current_clean = self._extract_content(file_path_current, pdf_type_current, mode, filter_footer=True)
            data_json_current_clean = data_json_current.copy()
        if pdf_type_future == "grid":
            data_json_future = self._extract_content(file_path_future, pdf_type_future, mode)
            data_json_future_clean = data_json_future.copy()
        elif pdf_type_future == "others":
            data_json_future = self._extract_content(file_path_future, pdf_type_future, mode)
            # data_json_future_clean = self._extract_content(file_path_future, pdf_type_future, mode, filter_footer=True)
            data_json_future_clean = data_json_future.copy()

        # Text Comparison
        common_lans = data_json_current.keys() & data_json_future.keys()

        if pdf_type_current == "grid":
            doc_current = get_doc(file_path_current,mode)
            grids_current = get_grid_bboxes(doc_current)
            symbols_legends_grid_current = get_symbol_legend_bboxes(doc_current, grids_current)
            grid_type_current = get_symbol_legend_language_type(doc_current)
            if grid_type_current == "multilingual":
                matches_current = [
                    [key, box]
                    for key, boxes in symbols_legends_grid_current.items()
                    for box in boxes
                ]
                legends_full_current = []
                for page_no, bbox in matches_current:
                    # for bbox in values:
                    legends_full_current.extend(get_symbol_descriptors(doc_current,page_no, bbox))

                img_path_current =  get_grid_imgs(doc_current, matches_current, mode)

        if pdf_type_future == "grid":
            doc_future = get_doc(file_path_future,mode)
            grids_future = get_grid_bboxes(doc_future)
            symbols_legends_grid_future = get_symbol_legend_bboxes(doc_future, grids_future)
            grid_type_future = get_symbol_legend_language_type(doc_future)
            if grid_type_future == "multilingual":
                matches_future = [
                    [key, box]
                    for key, boxes in symbols_legends_grid_future.items()
                    for box in boxes
                ]
                legends_full_future = []
                for page_no, bbox in matches_future:
                    # for bbox in values:
                    legends_full_future.extend(get_symbol_descriptors(doc_future,page_no, bbox))

                img_path_future =  get_grid_imgs(doc_future, matches_future, mode)
  
            
        for lan in common_lans:
            if lan == "en":
                response_subset = {}
                en_text_current = restructure_text(data_json_current_clean[lan]['text'], lan).splitlines()
                en_text_future = restructure_text(data_json_future_clean[lan]['text'], lan).splitlines()
                pairs = pair_strings_with_embeddings(en_text_current, en_text_future, max_concat=4)
                df = get_deviations(pairs)
                df['current (spelling error)'] = df['current'].apply(find_misspelled)
                df['future (spelling error)'] = df['future'].apply(find_misspelled)
                same = df['current (spelling error)'] == df['future (spelling error)']
                df.loc[same, ['current (spelling error)', 'future (spelling error)']] = "None"
                df["Comparison Status"] = df.apply(
                                                    lambda row: "Match" if row["deviation"] == "None" else "No Match", axis=1
                                                )
                df['Element Type'] = 'generic text'
                df_text_comparison = df[['Element Type', 'current', 'future', 'deviation','Comparison Status' , 'current (spelling error)',
                                        'future (spelling error)']]
                text_comparison_json = df_text_comparison.to_json(orient="records", force_ascii=False, indent=2)
                response_subset["text_comparison"] = text_comparison_json

                #structural Comparison
                comparison = {}
                response_subset["structural_analysis"] = {}
                current_ifu_text = data_json_current_clean[lan]['text']
                future_ifu_text = data_json_future_clean[lan]['text']

                df = compare_texts(current_ifu_text, future_ifu_text,'')
                df.drop('Translated GPT',axis=1, inplace=True)
                df.reset_index(inplace=True)
                df.rename(columns={'Original':"current", 'Transtaled Vendor':"future"}, inplace=True)
                df['Element Type'] = "special char"
                structural_comparison = df.to_json(orient="records", force_ascii=False, indent=2)
                response_subset["structural_analysis"]["structural_comparison"] = structural_comparison


                #table analysis

                table_report = {}
                if len(data_json_current[lan]['table']) == len(data_json_future[lan]['table']):
                    for i in range(len(data_json_current[lan]['table'])):
                        current_table = data_json_current[lan]['table'][i]['table']
                        future_table = data_json_future[lan]['table'][i]['table']
                        diff_df = create_difference_report_ifu_comp(current_table, future_table)
                        diff_df['Element Type'] = 'table'
                        table_report[i] = {
                            "table": diff_df.to_json(orient="records", force_ascii=False, indent=2),
                            "img_path_current": [data_json_current[lan]['table'][i]['image_path']],
                            "img_path_future": [data_json_future[lan]['table'][i]['image_path']],
                            "message": None
                        }

                    response_subset["structural_analysis"]["table_analysis"] = table_report

                elif len(data_json_current[lan]['table']) != len(data_json_future[lan]['table']):
                    min_tables = min(len(data_json_current[lan]['table']),
                                    len(data_json_future[lan]['table']))
                    comparison_made = False

                    for i in range(min_tables):
                        current_table = data_json_current[lan]['table'][i]['table']
                        future_table = data_json_future[lan]['table'][i]['table']

                        # Compare only when column counts match
                        current_cols = len(pd.DataFrame(current_table).columns)
                        future_cols = len(pd.DataFrame(future_table).columns)

                        if current_cols == future_cols:
                            diff_df = create_difference_report_ifu_comp(current_table, future_table)
                            diff_df['Element Type'] = 'table'
                            table_report[i] = {
                                "table": diff_df.to_json(orient="records", force_ascii=False, indent=2),
                                "img_path_current": [data_json_current[lan]['table'][i]['image_path']],
                                "img_path_future": [data_json_future[lan]['table'][i]['image_path']],
                                "message": None
                            }
                            comparison_made = True
                            response_subset["structural_analysis"]["table_analysis"] = table_report

                    # Tables exist on both sides but nothing was comparable
                    if min_tables > 0 and not comparison_made:
                        response_subset["structural_analysis"]["table_analysis"] = {"table": None,
                                                                                "img_path_current": None,
                                                                                "img_path_future": None,
                                                                                "message": f"The number of tables in current {len(data_json_current[lan]['table'])} do not match the number of tables in {len(data_json_future[lan]['table'])}. Check for split tables. "}

        # try:  
                try:
                    response_subset["symbol_legends_validation"] = {}
                    yolo_path = os.path.join(ROOT_DIR, 'object_detection', 'best.pt')
                    yolo_model = load_yolo_model(yolo_path)
                    empty_result = {"legends": '', "bbox": '', "img_path": ''}

                    # ---------- current ----------
                    if pdf_type_current == "grid":

                        if grid_type_current == "multilingual":
                            current_legends = [item[lan] for item in legends_full_current if lan in item]
                            symbol_detections_current = symbol_detect_comp(img_path_current, mode, yolo_model)
                        
                        else:
                            current_legends = []
                            metadata_grids = data_json_current[lan]["metadata"]["grids"]
                    
                            matches_current = common_bboxes(
                                symbols_legends_grid_current,
                                metadata_grids)
                            
                            img_path_current =  get_grid_imgs(doc_current, matches_current, mode)
                            for page_no, bbox in matches_current:
                                # for bbox in values:
                                current_legends.extend(get_symbol_descriptors(doc_current,page_no, bbox))
                            symbol_detections_current = symbol_detect_comp(img_path_current, mode, yolo_model)

            
            
                    elif pdf_type_current == "others":
                        doc = get_doc(file_path_current, mode)
                        current_legends = remove_en_prefix(extract_legends_others_ifu_comp(data_json_current, doc, {}, lan=lan))
                        image_paths = data_json_current[lan]['metadata']['image_path']
                        if not image_paths:
                            symbol_detections_current = empty_result
                        else:
                            if len(image_paths) > 1:
                                img_path = append_imgs(image_paths, mode, lan)
                            else:
                                img_path = image_paths[0]
                            symbol_detections_current = symbol_detect_comp(img_path, mode, yolo_model)

                    # ---------- future ----------
                    if pdf_type_future == "grid":
                        # doc = get_doc(file_path_future,mode)
                        if grid_type_future == "multilingual":
                            future_legends = [item[lan] for item in legends_full_future if lan in item]
                            symbol_detections_future = symbol_detect_comp(img_path_future, mode, yolo_model)
                        
                        else:
                            future_legends = []
                            metadata_grids = data_json_future[lan]["metadata"]["grids"]
                    
                            matches_future = common_bboxes(
                                symbols_legends_grid_future,
                                metadata_grids)
                            
                            img_path_future =  get_grid_imgs(doc_future, matches_future, mode)
                            for page_no, bbox in matches_future:
                                # for bbox in values:
                                future_legends.extend(get_symbol_descriptors(doc_future,page_no, bbox))

                            symbol_detections_future = symbol_detect_comp(img_path_future, mode, yolo_model)

                    elif pdf_type_future == "others":
                        doc = get_doc(file_path_future, mode)
                        future_legends = remove_en_prefix(extract_legends_others_ifu_comp(data_json_future, doc, {}, lan=lan))
                        image_paths = data_json_future[lan]['metadata']['image_path']
                        if not image_paths:
                            symbol_detections_future = empty_result
                        else:
                            if len(image_paths) > 1:
                                img_path = append_imgs(image_paths, mode,lan)
                            else:
                                img_path = image_paths[0]
                            symbol_detections_future = symbol_detect_comp(img_path, mode, yolo_model)


                    img_paths = {"current": symbol_detections_current["img_path"],
                            "future": symbol_detections_future["img_path"]}
                    response_subset["symbol_legends_validation"]["img_paths"] = img_paths

                    future_validation_mapper = symbol_detections_future.copy()

                    legend_pairs = pair_legends(current_legends, future_legends)
                    df_legends = get_deviations(legend_pairs)
                    # deviation_legends['Element Type'] = 'symbol'
                    # deviation_legends["Comparison Status (descriptor)"] = deviation_legends.apply(
                    #                                     lambda row: "Match" if row["deviation"] == "None" else "Match", axis=1
                    #                                 )
                    # deviation_legends_json = deviation_legends.to_json(orient="records", force_ascii=False, indent=2)
                    # self.response_json["symbol_legends_validation"]["legends_validation"] = deviation_legends_json


                    symbol_detections_current = sort_legends_by_layout(symbol_detections_current)
                    symbol_detections_future = sort_legends_by_layout(symbol_detections_future)

                    
                    paired_symbol_symbol_legends = pair_legends(symbol_detections_current["legends"], symbol_detections_future["legends"], threshold=0.8)
                    deviation_symbol_legends = get_deviations(paired_symbol_symbol_legends)
                    symbols = deviation_symbol_legends.copy()
                    # deviation_symbol_legends["Comparison Status (symbols)"] = deviation_symbol_legends.apply(
                    #                                     lambda row: "Match" if row["deviation"] == "None" else "Match", axis=1
                    #                                 )
                    deviation_symbols_json = deviation_symbol_legends.to_json(orient="records", force_ascii=False, indent=2)

                    paired_symbol_legends = pair_legends(symbol_detections_current["legends"], symbol_detections_future["legends"], threshold=0.8)
                    deviation_symbol_legends = get_deviations(paired_symbol_legends)

                    metadata_current_bucketed = bucket_metadata(symbol_detections_current, deviation_symbol_legends, 'current')
                    metadata_future_bucketed = bucket_metadata(symbol_detections_future, deviation_symbol_legends, 'future')
                            
                    deviation_symbols_json = deviation_symbol_legends.to_json(orient="records", force_ascii=False, indent=2)
                    response_subset["symbol_legends_validation"]["symbols_validation"] = {}
                    response_subset["symbol_legends_validation"]["symbols_validation"]["values"] = deviation_symbols_json
                    response_subset["symbol_legends_validation"]["symbols_validation"]['bbox'] = {"current": metadata_current_bucketed,
                                                                                                "future": metadata_future_bucketed}


                    if lan != "en":   
                        threshold = 0.5
                    else:
                        threshold = 0.6
        
                    translated_symbols_current = translate_symbols(self.llm_structured, symbol_detections_current['legends'], lan)
                    pairs_current_temp = pair_legends(df_legends["current"], translated_symbols_current, threshold)
                    pairs_current = restore_english_symbols(pairs_current_temp, symbol_detections_current['legends'], translated_symbols_current, symbol_key=1)

                    translated_symbols_future = translate_symbols(self.llm_structured, symbol_detections_future['legends'], lan)
                    pairs_future_temp = pair_legends(df_legends["current"], translated_symbols_future, threshold)
                    pairs_future = restore_english_symbols(pairs_future_temp, symbol_detections_future['legends'], translated_symbols_future, symbol_key=1)
                    # pairs_future = pair_legends(df_legends["future"], symbols["future"], threshold)

                    df_current = pd.DataFrame(pairs_current, columns = ["Descriptor (Current)", "Symbol Class (Current)"])
                    df_future = pd.DataFrame(pairs_future, columns = ["Descriptor (Future)", "Symbol Class (Future)"])
            
                    full_dataset = pd.concat([df_current, df_future], axis =1)
                    full_dataset = full_dataset.drop_duplicates(subset = ["Descriptor (Current)", "Descriptor (Future)"], keep="last")
                    full_dataset = full_dataset.reset_index(drop=True)
                    # full_dataset["Deviation (Descriptor)"] = get_deviations(full_dataset[["Descriptor (Current)", "Descriptor (Future)"]].values)['deviation']
                    # full_dataset["Deviation (Symbols)"] = get_deviations(full_dataset[["Descriptor (Current)", "Descriptor (Future)"]].values)['deviation']
                    full_dataset["Deviation (Descriptor)"] = full_dataset.apply(
                                                                        lambda r: describe_diff(r["Descriptor (Current)"], r["Descriptor (Future)"]), axis=1)

                    full_dataset["Deviation (Symbols)"] = full_dataset.apply(
                                                                lambda r: describe_diff(r["Symbol Class (Current)"], r["Symbol Class (Future)"]), axis=1)
                    full_dataset["Comparison Status (Descriptor)"] = full_dataset.apply(
                                                                lambda row: "Match" if row["Deviation (Descriptor)"] == "None" else "No Match", axis=1
                                                            )

                    full_dataset["Comparison Status (Symbol)"] = full_dataset.apply(
                                                                lambda row: "Match" if row["Deviation (Symbols)"] == "None" else "No Match", axis=1
                                                            )

                    full_dataset["Element Type"] = "symbol"
                    first_four = full_dataset.columns[:4]

                    # Keep rows where NOT all of the first 4 columns are empty strings
                    full_dataset = full_dataset[~(full_dataset[first_four] == "").all(axis=1)].reset_index(drop=True)
                    

                    full_dataset = add_bbox_columns(full_dataset, response_subset["symbol_legends_validation"]["symbols_validation"])
                    
                    full_dataset = add_future_value_validation(full_dataset, future_validation_mapper)

                    if lan == "zh-hans":
                        cols = ["Symbol Class (Current)", "Symbol Class (Future)"]

                        both_blank = full_dataset[cols].replace(r"^\s*$", pd.NA, regex=True).isna().all(axis=1)
                        full_dataset = full_dataset.loc[~both_blank].reset_index(drop=True)

                    response_subset["symbol_legends_validation"]["full_comparison"] = full_dataset.to_json(orient="records", force_ascii=False, indent=2)
            
                except:
                    response_subset["symbol_legends_validation"] = {}
                ## Datamatrix validation

                doc_current = get_doc(file_path_current, mode)
                doc_future = get_doc(file_path_future, mode)
                datamatrix_metadata_current, img_path_current_datamatrix = get_datamatrix_ifu(doc_current,pdf_type_current, mode, yolo_model)
                datamatrix_metadata_future, img_path_future_datamatrix = get_datamatrix_ifu(doc_future,pdf_type_future, mode, yolo_model)

                if datamatrix_metadata_current is not None:
                    datamatrix_metadata_current = datamatrix_metadata_current[0]

                if datamatrix_metadata_future is not None:
                    datamatrix_metadata_future = datamatrix_metadata_future[0]
                
                datamatrix_grading = make_comp_dict_datamatrix(datamatrix_metadata_current, datamatrix_metadata_future)
                datamatrix_validation = {"Element Type": "Barcode",
                                    "barcode_grading": datamatrix_grading,
                                "img_path": {"current":img_path_current_datamatrix,
                                            "future": img_path_future_datamatrix}}


                response_subset['barcode validation'] = datamatrix_validation 

                # self.save_results()

                self.response_json[lan] = response_subset

        return self.response_json

            
    
    def _extract_content(self, pdf_path, pdf_type, mode, extract_images=True,filter_footer=False):
        """Handles local or blob storage extraction."""
        if mode == "blob":
            return extract_data(pdf_path, pdf_type, self.llm_structured,self.llm, self.client_ocr,self.yolo_model, extract_images =extract_images, mode="blob",filter_footer=filter_footer)
        
        return extract_data(pdf_path, pdf_type, self.llm_structured,self.llm,  self.client_ocr,self.yolo_model, extract_images =extract_images, mode="local",filter_footer=filter_footer)


    def save_results(self, output_path="ifu_comp_output.json"):
            """Saves the processed data to a JSON file."""
            with open(output_path, "w") as f:
                json.dump(self.response_json, f, indent=4, cls=NumpyEncoder)