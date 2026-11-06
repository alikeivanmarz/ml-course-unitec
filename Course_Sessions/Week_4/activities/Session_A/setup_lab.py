"""Explicit notebook setup for a local Ollama service or a lecturer endpoint.

Importing this file installs nothing, starts nothing and downloads nothing.
prepare_lab(download_missing=True) permits local model downloads only.
"""
import json
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parent
INSTALL_URL = "https://ollama.com/download"
CLASS_SERVER_URL = "https://ml-class.roadinsite.co.nz"
# Class server token for this semester. The lecturer updates both lines each semester.
CLASS_TOKEN = "js4m-gnvv-4fes"
CLASS_TOKEN_EXPIRES = "2026-11-08"
CLIENT_NAME = "Week4-Class-Lab/1.0"
_BUILT_IN_TOKEN_REFUSED = False
_SERVER_PROCESS = None


def _local(endpoint):
    return urlparse(endpoint).hostname in {"localhost", "127.0.0.1", "::1"}


def _opener(endpoint):
    return build_opener(ProxyHandler({})) if _local(endpoint) else build_opener()


def _request(endpoint, route, payload=None, timeout=5):
    headers = {"Content-Type": "application/json", "User-Agent": CLIENT_NAME}
    if os.environ.get("LLM_LAB_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["LLM_LAB_TOKEN"]
    return _opener(endpoint).open(Request(endpoint + route, headers=headers,
        data=None if payload is None else json.dumps(payload).encode()), timeout=timeout)


def _json(endpoint, route):
    with _request(endpoint, route) as response:
        return json.load(response)


def _tags(endpoint):
    value = _json(endpoint, "/api/tags")
    if not isinstance(value, dict) or not isinstance(value.get("models"), list):
        raise ValueError("This address did not return an Ollama model list. Check the model-service URL.")
    return [item["name"] for item in value["models"]
            if isinstance(item, dict) and isinstance(item.get("name"), str)]


def _binary_candidates(system=None, environ=None, home=None):
    system = system or platform.system()
    environ = os.environ if environ is None else environ
    home = Path.home() if home is None else Path(home)
    if system == "Darwin":
        return [Path("/Applications/Ollama.app/Contents/Resources/ollama"),
                home / "Applications/Ollama.app/Contents/Resources/ollama"]
    if system == "Windows":
        return [Path(environ[key]) / suffix for key, suffix in [
            ("LOCALAPPDATA", "Programs/Ollama/ollama.exe"),
            ("PROGRAMFILES", "Ollama/ollama.exe")]
            if environ.get(key)]
    return []


def _find_ollama():
    command = shutil.which("ollama")
    if command:
        return command
    return next((str(p) for p in _binary_candidates() if p.is_file()), None)


def _start_local(command, endpoint):
    global _SERVER_PROCESS
    if not _local(endpoint):
        raise ValueError("The notebook starts services only on this laptop's loopback address.")
    log = ROOT / "logs/ollama-setup.log"
    log.parent.mkdir(exist_ok=True)
    options = {"stdin": subprocess.DEVNULL, "stderr": subprocess.STDOUT,
               "env": {**os.environ, "OLLAMA_HOST": endpoint}}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    with log.open("ab") as stream:
        _SERVER_PROCESS = subprocess.Popen([command, "serve"], stdout=stream, **options)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            return _tags(endpoint)
        except HTTPError:
            raise
        except (URLError, OSError):
            if _SERVER_PROCESS.poll() is not None:
                break
            time.sleep(0.5)
    raise RuntimeError("Ollama did not start. Open the Ollama app and rerun this cell. "
                       "If it still fails, show logs/ollama-setup.log to the lecturer.")


def _model_key(name):
    return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"


def _missing(required, available):
    present = {_model_key(name) for name in available}
    return [name for name in required if _model_key(name) not in present]


def _pull(endpoint, name):
    if not _local(endpoint):
        raise ValueError("Students must not download models to a lecturer's server.")
    print("Downloading:", name, "(leave this cell running)")
    finished = False
    previous = None
    with _request(endpoint, "/api/pull", {"model": name, "stream": True}, timeout=120) as response:
        for raw in response:
            if not raw.strip():
                continue
            item = json.loads(raw)
            if item.get("error"):
                raise RuntimeError(str(item["error"])[:300])
            status = item.get("status", "Working")
            total = item.get("total", 0)
            percent = int(100 * item.get("completed", 0) / total) if total else None
            progress = (status, None if percent is None else percent // 10)
            if progress != previous:
                print(" ", status, "" if percent is None else f"{percent}%")
                previous = progress
            finished = status == "success"
    if not finished:
        raise RuntimeError("The download ended before success. Check your connection and rerun the setup cell.")


def _stop(stage, message):
    print("\nACTION NEEDED:", message)
    print("Stop before the first live model request. Complete this step, then rerun the preparation cell.")
    return {"ready": False, "stage": stage, "message": message}


def _installation_help():
    message = ("Install Ollama once from " + INSTALL_URL + ". Windows: run the installer and open Ollama "
               "from Start. Mac: drag Ollama into Applications, then open it. Return here and rerun this cell.")
    try:
        from IPython import get_ipython
        from IPython.display import Markdown, display
        if get_ipython() is not None:
            display(Markdown("[Download Ollama](https://ollama.com/download) | "
                             "[Windows guide](https://docs.ollama.com/windows) | "
                             "[Mac guide](https://docs.ollama.com/macos)"))
    except ImportError:
        pass
    return _stop("installation", message)


def _ask_class_token():
    """Use the built-in class token; ask for one only if the server refused it."""
    if os.environ.get("LLM_LAB_TOKEN"):
        return
    if not _BUILT_IN_TOKEN_REFUSED and CLASS_TOKEN:
        os.environ["LLM_LAB_TOKEN"] = CLASS_TOKEN
        return
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            return
    except ImportError:
        return
    try:
        token = input("Class token from the lecturer's screen (for example abcd-1234-wxyz): ")
    except (EOFError, KeyboardInterrupt):
        return
    token = "".join(token.split()).lower()
    if token:
        os.environ["LLM_LAB_TOKEN"] = token


def _server_message(error):
    try:
        return json.loads(error.read(1000).decode(errors="replace")).get("error", "")
    except (ValueError, AttributeError, OSError):
        return ""


def prepare_lab(service="local", class_url="", download_missing=False):
    """Return a clear readiness result; expected setup problems do not show a traceback.

    service='local' uses this laptop. service='class' uses the lecturer's class
    server (CLASS_SERVER_URL), asks for the class token in a notebook, and never
    starts a service or downloads weights remotely.
    """
    import llm_lab
    if service not in {"local", "class"}:
        return _stop("configuration", 'Choose service="local" or service="class".')
    endpoint = "http://localhost:11434" if service == "local" else class_url.strip().rstrip("/")
    try:
        parsed = urlparse(endpoint)
    except ValueError:
        return _stop("configuration", "The model-service URL is malformed. Copy the actual URL supplied by the lecturer.")
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        return _stop("configuration", "Enter the actual HTTP(S) model-service URL supplied by the lecturer. "
                     "Keep tokens in LLM_LAB_TOKEN, not in the URL or saved code.")
    if service == "class" and _local(endpoint):
        return _stop("configuration", "localhost means this laptop. Ask for the lecturer's reachable server URL, "
                     "or choose local mode for this laptop.")
    os.environ["OLLAMA_HOST"] = endpoint
    if service == "class":
        _ask_class_token()
    required = list(dict.fromkeys([llm_lab.TEXT_MODEL, llm_lab.VISION_MODEL, llm_lab.EMBED_MODEL]))
    print("Python:", platform.python_version(), "| System:", platform.system())
    print("Mode:", service, "| Required models:", ", ".join(required))
    try:
        try:
            available = _tags(endpoint)
            print("Model service: connected")
        except HTTPError:
            raise
        except (URLError, OSError):
            if service == "class":
                return _stop("connection", "The class server cannot be reached. Check your internet connection and "
                             "that CLASS_URL is " + CLASS_SERVER_URL + ". The server runs only during class; "
                             "outside class use SERVICE = \"local\".")
            command = _find_ollama()
            if not command:
                return _installation_help()
            print("Starting the installed local Ollama service...")
            available = _start_local(command, endpoint)
        try:
            information = _json(endpoint, "/api/version")
            version = information.get("version", "unknown") if isinstance(information, dict) else "unknown"
        except (HTTPError, URLError, ValueError):
            version = "unknown"
        print("Service version:", version)
        numbers = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(version))
        if (numbers and tuple(map(int, numbers.groups())) < (0, 12, 7)
                and any(name.startswith("qwen3-vl:") for name in required)):
            return _stop("version", "Qwen3-VL needs Ollama 0.12.7 or newer. "
                         + ("Install the current Ollama app, then reopen it." if service == "local"
                            else "Ask the lecturer to update the class server."))
        missing = _missing(required, available)
        if missing and service == "class":
            return _stop("models", "The class server is missing: " + ", ".join(missing)
                         + ". Ask the lecturer to supply these names or the configured alternatives.")
        if missing and not download_missing:
            return _stop("download", "Missing models: " + ", ".join(missing)
                         + ". When you have internet and several GB free, set DOWNLOAD_MISSING_MODELS = True and rerun.")
        for name in missing:
            _pull(endpoint, name)
        available = _tags(endpoint)
        if _missing(required, available):
            return _stop("models", "A required model is still missing. Rerun setup and inspect the download messages.")
        print("\nREADY: the service is connected and all required model names are available.")
        print("Next: run the first language-model request. Check the answer, not just whether the cell finishes.")
        return {"ready": True, "stage": "ready", "service": service,
                "available": available, "required": required, "version": version}
    except HTTPError as error:
        if error.code in {401, 403}:
            detail = _server_message(error)
            if service == "class":
                global _BUILT_IN_TOKEN_REFUSED
                built_in = os.environ.pop("LLM_LAB_TOKEN", None) == CLASS_TOKEN
                if built_in:
                    _BUILT_IN_TOKEN_REFUSED = True
                    return _stop("authentication", (detail or "The class server refused the built-in token.")
                                 + " Pull the latest course files (git pull origin main), restart the kernel and "
                                 "run from the top. Or rerun this cell and type the token on the lecturer's screen.")
                return _stop("authentication", (detail or "The class server refused the token.")
                             + " Rerun this cell and type the token exactly as shown on the lecturer's screen.")
            return _stop("authentication", "The service refused access. Ask the lecturer about endpoint permissions "
                         "and LLM_LAB_TOKEN. Do not put a token in submitted notebook cells.")
        if service == "class" and error.code in {502, 503, 504, 530}:
            return _stop("connection", "The class server is not running right now. It runs only during class. "
                         "Tell the lecturer, or outside class use SERVICE = \"local\".")
        return _stop("service", f"The service returned HTTP {error.code}. Check the URL or update Ollama. "
                     "If this happened during a download, check internet access and free storage, then rerun setup.")
    except (URLError, OSError, ValueError, RuntimeError) as error:
        return _stop("setup", str(error)[:400] + " Check internet access and free storage. "
                     "For a blocked installation, ask the lecturer about class-server access.")
