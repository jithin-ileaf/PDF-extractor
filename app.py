from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import zipfile, io, os, re, json, threading, uuid, warnings, tempfile
from typing import Dict, List
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
from paddleocr import PaddleOCR
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

# -------------------------
# Configure LLM (Gemini)
# -------------------------
genai.configure(api_key=os.getenv("API_KEY"))
model = genai.GenerativeModel(
    model_name="gemini-2.5-pro",
    generation_config={
        "temperature": 0.2,
        "response_mime_type": "application/json",
    },
)

# -------------------------
# Initialize OCR once
# -------------------------
ocr = PaddleOCR(
    lang='en',
    use_angle_cls=True,
    text_det_thresh=0.2,
    text_det_box_thresh=0.3,
    text_det_unclip_ratio=1.7,
)

# -------------------------
# Globals + Thread-safety
# -------------------------
app = FastAPI()
state_lock = threading.Lock()

PROCESS_STATE = {
    "id": None,
    "status": "idle",        # idle | processing | completed | error | stopped
    "current_file": None,
    "processed": 0,
    "total": 0,
    "errors": [],
}

processed_results: Dict[str, Dict] = {}
pdf_file_list: List[str] = []
uploaded_zip_bytes: bytes = b""
STOP_PROCESSING = False  # flag to stop processing

# -------------------------
# Helpers
# -------------------------

def sort_key(filename: str):
    master_match = re.search(r'master(\d+)', filename)
    master_num = int(master_match.group(1)) if master_match else 0
    variant_match = re.search(r'master\d+-(\d+)', filename)
    variant_num = int(variant_match.group(1)) if variant_match else -1
    return (master_num, variant_num)

def compact_coordinates(json_text: str) -> str:
    return re.sub(r"\[\s+([\d\.\,\s\-eE]+?)\s+\]", lambda m: '[' + ' '.join(m.group(1).split()) + ']', json_text)

# -------------------------
# Core processing functions
# -------------------------

def run_llm_extract_fields(txt_text: str,extra_data: Dict[str, str]) -> str:
    prompt = f"""
            You are given a full text extracted from a legal document.

            <text> : {txt_text}

            You need to carefully analyse the text and extract values for the following keys based on the context provided.
            {'\n'.join(f"   {key}: {value}" for key, value in extra_data.items())}
            Extract values for the above mentioned keys only in the below given json format
            <output_format>
            {{"key": {{
            "Extracted Value": "",
                    }}
            }}
            - Finally add a new key called 'Amendment Status' in the following format
                {{"Amendment Status": {{
                "Extracted Value": {{
                "Is Amendment": Whether the document is a base contract or an amendment/cancellation, output either "True" or "False".
                "Amendment Details": {{
                    "Base Contract ID/Number": Contract id/number of the base contract,
                    "Base Contract Title": Title of the base contract,
                    "Base Contract Date": Date of the base contract
                }}
                }}
                }}
                }}

            RULES TO BE SRICTLY FOLLOWED:
            - Extracts values only for the keys mentioned above.
            - Extract accurate values for the keys listed strictly based on the provided context. DO NOT TRY TO FILL IN WITH WRONG TEXT
            - **Do not rephrase or create new text. Extract values based on the context as it appeared in the input.**
            - For any key if the value doesn't exist then provide in the below format
                "key": {{
                    "Extracted Value": "Not specified",
                }}
            
            **Ensure all the rules are strictly followed before providing the output**.

                """  
    response = model.generate_content([
        {"role": "model", "parts": "You are a specialist in analysing a legal document and extracting keys and its values based on a given context"},
        {"role": "user", "parts": prompt},
    ])
    return response.text

def run_llm_add_positions(llm_json_str: str, ocr_list: List[Dict]) -> str:
    prompt_2 = f"""
                You are given a json string along with a list containing some dictionaries.
                <json_string> : {llm_json_str}
                <list> : {json.dumps(ocr_list)}

                The json string contains information in the below format -
                {{"key": {{
                "Extracted Value": ""
                        }}
                }}
                Each dictionary in the above list contains the "text" and its "position" extracted from a legal document in the below given format.
                <input_format>
                [{{'text': ' ', 'position': {{'page': 1, 'coordinates': [[x1, y1],[x2,y2],[x3,y3],[x4,y4]]}}}},...]
                Note : Here the coordinates are in the order - [top-left],[top-right],[bottom-right],[bottom-left]

                Your job is to 
            - Find the position of texts under "Extracted Value" using the dictionaries in the list and re-calcualte the coordinates to create a new bounding box.
            - Append the position, page and coordinate values to the key without changing the text.
            - If the text is not present or mentioned as "Not specified"  then provide in the below format
                "Extracted Value": "Not specified",
                "Position": "Not specified"
            - Do not create position information for "Agreement summary" key.
            - Follow the below format for "Amendment Status" key.
                {{"Amendment Status": {{
                "Extracted Value": {{
                "Is Amendment": Whether the document is a base contract or an amendment/cancellation. Output "True" if there is a clear indication of amendment. Else  output "False".
                "Amendment Details": {{
                    "Base Contract ID/Number": contract id/number of the referenced contract,
                    "Base Contract Title": title of the referenced contract,
                    "Base Contract Date": date of the referenced contract
                }}
                }},
                "Position": {{
                    "Page": 5,
                    "Coordinates": [
                        [0.075, 0.603],
                        [1.01, 0.603],
                        [1.01, 0.708],
                        [0.075, 0.678]
                                    ]
            }}
                }}
                }}
            - If the position field is not present then ignore it.Do not add position field and try to fill it.

            <expected output format>
            {{"key": {{
            "Extracted Value": "",
            "Position": {{
            "Page": ,
            "Coordinates": [
                [_,_],
                [_,_],
                [_,_],
                [_,_]
            ]
                        }}
                    }}
            }}

            **Ensure all the rules are strictly followed before providing the output**
            """  
    response2 = model.generate_content([
        {"role": "model", "parts": "You are a specialist in extracting coordinates of a given text and creating a bounding box."},
        {"role": "user", "parts": prompt_2},
    ])
    return response2.text

def ocr_pdf_and_extract(zip_bytes: bytes, file_in_zip: str, tmp_dir: str) -> (List[Dict], str):
    pdf_bytes = None
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        pdf_bytes = zf.read(file_in_zip)

    base_name = Path(file_in_zip).name
    file_stem = Path(base_name).stem
    temp_pdf_path = os.path.join(tmp_dir, f"{file_stem}.pdf")
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_bytes)

    pages = convert_from_path(temp_pdf_path, dpi=100)
    ocr_list: List[Dict] = []
    combined_texts: List[str] = []

    try:
        for page_num, page in enumerate(pages, start=1):
            img_path = os.path.join(tmp_dir, f"{file_stem}_page_{page_num}.jpg")
            page.save(img_path, "JPEG")
            image = Image.open(img_path)
            width, height = image.size

            result = ocr.predict(img_path)
            rec_texts = result[0]['rec_texts']
            rec_boxes = [arr.tolist() for arr in result[0]['rec_boxes']]

            for i, txt in enumerate(rec_texts):
                box = rec_boxes[i]
                coords = [
                    [round(box[0] / width, 3), round(box[1] / height, 3)],
                    [round(box[2] / width, 3), round(box[1] / height, 3)],
                    [round(box[2] / width, 3), round(box[3] / height, 3)],
                    [round(box[0] / width, 3), round(box[3] / height, 3)],
                ]
                ocr_list.append({
                    "text": txt,
                    "position": {"page": page_num, "coordinates": coords}
                })
                combined_texts.append(txt)
            try:
                os.remove(img_path)
            except Exception:
                pass
    finally:
        try:
            os.remove(temp_pdf_path)
        except Exception:
            pass

    txt_text = " ".join(combined_texts)
    return ocr_list, txt_text

def process_single_pdf(zip_bytes: bytes, file_in_zip: str, extra_data: Dict) -> Dict:
    with tempfile.TemporaryDirectory() as tmp_dir:
        ocr_list, txt_text = ocr_pdf_and_extract(zip_bytes, file_in_zip, tmp_dir)
        llm_response_text = run_llm_extract_fields(txt_text,extra_data)
        llm_with_pos_text = run_llm_add_positions(llm_response_text, ocr_list)

        try:
            final_data = json.loads(llm_with_pos_text)
        except Exception:
            final_data = {"llm_raw": llm_with_pos_text, "llm_extracted": llm_response_text}

        final_json_text = json.dumps(final_data)
        final_json_text = compact_coordinates(final_json_text)
        try:
            final_data = json.loads(final_json_text)
        except Exception:
            pass

        return final_json_text

def background_process_all_pdfs(process_id: str,extra_data: Dict):
    global PROCESS_STATE, processed_results, pdf_file_list, uploaded_zip_bytes, STOP_PROCESSING
    try:
        with state_lock:
            PROCESS_STATE['status'] = 'processing'
            PROCESS_STATE['current_file'] = None
            PROCESS_STATE['processed'] = 0
            PROCESS_STATE['total'] = len(pdf_file_list)

        for file_in_zip in pdf_file_list:
            if STOP_PROCESSING:
                with state_lock:
                    PROCESS_STATE['status'] = 'stopped'
                break

            file_display_name = Path(file_in_zip).name
            with state_lock:
                PROCESS_STATE['current_file'] = file_display_name

            try:
                final_data = process_single_pdf(uploaded_zip_bytes, file_in_zip, extra_data)
                with state_lock:
                    processed_results[file_display_name] = final_data
                    PROCESS_STATE['processed'] += 1
            except Exception as e:
                err_msg = f"Error processing {file_display_name}: {repr(e)}"
                with state_lock:
                    PROCESS_STATE['errors'].append(err_msg)
            finally:
                with state_lock:
                    PROCESS_STATE['current_file'] = None

        with state_lock:
            if PROCESS_STATE['status'] != 'stopped':
                PROCESS_STATE['status'] = 'completed'
    except Exception as e:
        with state_lock:
            PROCESS_STATE['status'] = 'error'
            PROCESS_STATE['errors'].append(repr(e))

# -------------------------
# API Endpoints
# -------------------------

@app.post('/upload-zip')
async def upload_zip(zip_file: UploadFile = File(...),
    data: str = Form(default="{}")   # accept JSON as string
):
    global uploaded_zip_bytes, pdf_file_list, processed_results, PROCESS_STATE, STOP_PROCESSING,extra_data

    try:
        extra_data = json.loads(data) if data else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail='Invalid JSON in data parameter')

    uploaded_zip_bytes = await zip_file.read()

    try:
        with zipfile.ZipFile(io.BytesIO(uploaded_zip_bytes)) as zf:
            files = [f for f in zf.namelist() if f.lower().endswith('.pdf')]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Invalid zip file: {repr(e)}')

    if not files:
        raise HTTPException(status_code=400, detail='No PDF files found in the uploaded zip')

    files = sorted(files, key=sort_key)
    files = files[1:2]  # Limit to first 50 PDFs
    with state_lock:
        process_id = str(uuid.uuid4())
        PROCESS_STATE = {
            'id': process_id,
            'status': 'idle',
            'current_file': None,
            'processed': 0,
            'total': len(files),
            'errors': [],
        }
        processed_results.clear()
        STOP_PROCESSING = False  # reset stop flag

    pdf_file_list = files

    worker = threading.Thread(target=background_process_all_pdfs, args=(process_id,extra_data), daemon=True)
    worker.start()

    return JSONResponse({
        'message': 'Processing started',
        'process_id': process_id,
        'total_pdfs': len(files)
    })

@app.post('/stop')
def stop_processing():
    """Stop the ongoing PDF processing."""
    global STOP_PROCESSING
    with state_lock:
        if PROCESS_STATE['status'] not in ('processing', 'idle'):
            return JSONResponse({'message': 'No active processing to stop', 'status': PROCESS_STATE['status']})
        STOP_PROCESSING = True
    return JSONResponse({'message': 'Processing will be stopped shortly'})

@app.get('/status')
def get_status():
    with state_lock:
        return JSONResponse(PROCESS_STATE)

@app.get('/results')
def list_results():
    with state_lock:
        processed_list = list(processed_results.keys())
        return JSONResponse({'processed_files': processed_list, 'processed_count': len(processed_list)})

@app.get('/results/{file_name}')
def get_result(file_name: str):
    with state_lock:
        if file_name not in processed_results:
            for k in processed_results.keys():
                if Path(k).name == file_name:
                    return JSONResponse({"file_name": Path(k).name, "extracted_data": processed_results[k]})
            if PROCESS_STATE['status'] in ('processing', 'idle'):
                return JSONResponse({'message': 'File not processed yet or does not exist', 'file_name': file_name}, status_code=404)
            else:
                return JSONResponse({'message': 'File not processed or does not exist', 'file_name': file_name}, status_code=404)

        return JSONResponse({"file_name": file_name, "extracted_data": processed_results[file_name]})
