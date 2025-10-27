import os
import shutil
import subprocess
import tempfile
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from starlette.responses import JSONResponse

from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ.get('GOOGLE_API_KEY'))
app = FastAPI(title="YouTube Snapshot -> Gemini", version="0.1.0")

CAMERAS = {
    "Flat bridge camera": "https://www.youtube.com/live/k4Lt_iev8x4?si=45kCzsruYXUddRzL",
    "Half way tree camera": "https://www.youtube.com/live/RHkdQI2PSKA?si=X0__U9WhRzjW0JCG",
    "Cross Road camera": "https://www.youtube.com/live/jJ6C03WtBJE?si=vrh6gPZ_2TQxMFIU",
    "Downtown parade camera": "https://www.youtube.com/live/u70ySp4OuHY?si=DevaOxmgThDS7SLb",
    "Devon house camera": "https://www.youtube.com/live/pmzJ4AAh0Ds?si=IolpuEHkLBCz2mVB",
    "Kingston Harbor camera": "https://www.youtube.com/live/jVr7_V4Tohw?si=wj2BAx2ior9cRJRb",
    "Westmoreland Little London": "https://www.youtube.com/live/SPwW9xN3e1M?si=mUL_BKaNMOdRdtgz",
    "Barbican camera": "https://www.youtube.com/live/I0w-636mEDY?si=A9FAO9p_pAIf6GRE"
}


class DescribeRequest(BaseModel):
    duration: Optional[int] = Query(15, ge=5, le=60)  # seconds to record (5-60)
    prompt: Optional[str] = "Describe what is happening visually and narratively in this clip."


def select_camera_url(query: str) -> str | None:
    """
    Uses an LLM call to find the best camera URL for a given query.
    """
    print(f"Routing query: '{query}'")
    
    # Create a list of camera names for the prompt
    camera_names = list(CAMERAS.keys())
    
    # This prompt asks the LLM to act as a router
    prompt = f"""
    You are a camera routing assistant. A user's query is: "{query}"

    Which of the following locations is the user asking about?
    
    Locations:
    {camera_names}
    
    Respond with *only* the name of the best matching location from the list.
    If no location matches well, respond with the single word "NONE".
    """
    
    try:
        response = client.models.generate_content(
                  model="gemini-2.5-flash",
                  contents=prompt,
        )
        selected_location = response.text.strip().strip("'\"") # Clean up quotes
        
        print(f"Router selected: {selected_location}")
        
        # Look up the URL in our dictionary
        return CAMERAS.get(selected_location) # Returns None if not found
        
    except Exception as e:
        print(f"Error during camera selection: {e}")
        return None

def run(cmd, capture_output=False, text=True, check=True):
    """Run subprocess helper."""
    result = subprocess.run(cmd, capture_output=capture_output, text=text)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nReturn code: {result.returncode}\n"
            f"Stdout: {result.stdout}\nStderr: {result.stderr}"
        )
    return result


def get_stream_url(youtube_url: str) -> str:
    """
    Uses yt-dlp -g to get the direct stream URL.
    Raises RuntimeError on failure.
    """
    cmd = ["yt-dlp", "-f", "best", "-g", youtube_url]
    try:
        res = run(cmd, capture_output=True, text=True)
    except Exception as e:
        raise RuntimeError(f"yt-dlp failed to get stream URL: {e}")
    stream_url = res.stdout.strip()
    if not stream_url:
        raise RuntimeError("No stream URL returned by yt-dlp")
    return stream_url


def record_stream_to_file(stream_url: str, duration: int, out_path: str):
    """
    Uses ffmpeg to record `duration` seconds from stream_url to out_path.
    """
    # -y overwrite, -t duration seconds, -c copy to avoid re-encoding when possible
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        stream_url,
        "-t",
        str(duration),
        "-c",
        "copy",
        out_path,
    ]
    try:
        run(cmd, capture_output=True)
    except Exception as e:
        # If -c copy fails due to codec/stream issues, fallback to re-encode baseline
        # Re-encode fallback (slower but more robust)
        fallback_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            stream_url,
            "-t",
            str(duration),
            "-vf",
            "scale=1280:-2",  # keep width reasonable
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            out_path,
        ]
        run(fallback_cmd, capture_output=True)


def analyze_with_gemini(video_path: str, prompt: str) -> str:
    try:
        video_bytes = open(file, 'rb').read()

        response = client.models.generate_content(
            # model='models/gemini-2.0-flash',
            model="gemini-2.5-flash",
            contents=types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                    ),
                    types.Part(text=query)
                ]
            )
        )
        return response.text
        

    except Exception as e:
        print(f"An error occurred during video analysis: {e}")
        raise


@app.post("/describe")
def describe(req: DescribeRequest):
    duration = int(req.duration)
    prompt = req.prompt or "Describe what is happening."


    # create a temp directory to stage files
    tmpdir = tempfile.mkdtemp(prefix="yt_snap_")
    out_file = os.path.join(tmpdir, f"clip_{uuid.uuid4().hex}.mp4")

    try:
        youtube_url = select_camera_url(prompt)
        if not youtube_url:
            raise HTTPException(status_code=400, detail="No camera selected")
        # 1) get direct stream url
        stream_url = get_stream_url(youtube_url)

        # 2) record N seconds to a file
        record_stream_to_file(stream_url, duration, out_file)

        # check file exists and non-empty
        if not os.path.exists(out_file) or os.path.getsize(out_file) < 1024:
            raise RuntimeError("Recorded file missing or too small")

        # 3) analyze with Gemini (or whichever model)
        analysis_text = analyze_with_gemini(out_file, prompt)

        # 4) return result
        return JSONResponse({"ok": True, "description": analysis_text})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # cleanup
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


@app.get("/health")
def health():
    return {"ok": True, "service": "yt-snapshot-gemini", "version": "0.1.0"}
