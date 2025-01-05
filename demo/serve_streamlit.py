import shlex
import subprocess
from pathlib import Path
import modal

streamlit_script_local_path = Path(__file__).parent / "app.py"
streamlit_script_remote_path = "/root/app.py"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("streamlit~=1.35.0", "evaluate", "transformers", "torch", "unsloth", "psutil")
    .add_local_file(
        streamlit_script_local_path,
        streamlit_script_remote_path,
    )
)

app = modal.App(name="superfloat-quantizer-streamlit", image=image)

if not streamlit_script_local_path.exists():
    raise RuntimeError(
        "app.py not found! Place the script with your streamlit app in the same directory."
    )

@app.function(
    allow_concurrent_inputs=100,
    gpu="A100",  # Specify the GPU type (e.g., "A100", "H100")
    timeout=86400,  # Timeout in seconds (1 day = 86400 seconds)

)
@modal.web_server(8000)
def run():
    target = shlex.quote(streamlit_script_remote_path)
    cmd = f"streamlit run {target} --server.port 8000 --server.enableCORS=false --server.enableXsrfProtection=false"
    subprocess.Popen(cmd, shell=True)