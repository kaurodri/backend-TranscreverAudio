from fastapi import FastAPI, File, UploadFile
import whisper
import os
import uvicorn
from fastapi.responses import JSONResponse

app = FastAPI()

model = whisper.load_model("base")

@app.get("/")
def read_root():
    return {"message": "backend-TranscreverAudios"}

@app.post("/transcrever/")
async def transcreverAudio(file: UploadFile = File(...)):
    try:
        #salva o arquivo temporariamente:
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as f:
            f.write(await file.read())
        
        result = model.transcribe(temp_file)

        #remove o arquivo temporário:
        os.remove(temp_file)
        
        return {"text": result["text"]}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
