# sip_incoming_llm_best_turns.py
# End-to-end: Twilio SIP -> PJSUA2 -> OpenAI Realtime (speech) -> PJSUA2 -> Twilio.
# Fluent, turn-based conversation: server-side VAD creates replies automatically.

import os, time, threading, traceback, json, base64, asyncio, re, gc
import logging
from dotenv import load_dotenv
import numpy as np
import pjsua2 as pj
from twilio.rest import Client
import websockets
from collections import deque
from datetime import datetime

# ========= Env =========
load_dotenv()

ACCT_SID  = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOK  = os.getenv("TWILIO_AUTH_TOKEN")
FROM_NUM  = os.getenv("TWILIO_PHONE_NUMBER")
TO_NUM    = os.getenv("PRACTICE_PHONE")

DOMAIN    = os.getenv("TWILIO_SIP_DOMAIN")     # left part only
USER      = os.getenv("TWILIO_SIP_USERNAME")   # SIP Credential List username
PASS      = os.getenv("TWILIO_SIP_PASSWORD")   # SIP Credential List password
OAI_KEY   = os.getenv("OPENAI_API_KEY")

# Call scheduling
TIMEZONE  = os.getenv("TIMEZONE", "UTC")
CALL_TIME = os.getenv("CALL_TIME", "NOW")

def check_environment():
    """Check environment variables - called after logging setup"""
    for k, v in {
        "TWILIO_ACCOUNT_SID": ACCT_SID, "TWILIO_AUTH_TOKEN": AUTH_TOK,
        "TWILIO_PHONE_NUMBER": FROM_NUM, "PRACTICE_PHONE": TO_NUM,
        "TWILIO_SIP_DOMAIN": DOMAIN, "TWILIO_SIP_USERNAME": USER,
        "TWILIO_SIP_PASSWORD": PASS, "OPENAI_API_KEY": OAI_KEY
    }.items():
        if not v:
            # This will now go to the log file instead of console
            raise RuntimeError(f"Missing {k} in environment/.env")

DOMAIN_FQDN   = f"{DOMAIN}.sip.twilio.com"
REGISTRAR_URI = f"sip:{DOMAIN_FQDN}"
AOR_URI       = f"sip:{USER}@{DOMAIN_FQDN}"

# ========= Globals / sync flags =========
incoming_call = None
media_ready   = threading.Event()
done_event    = threading.Event()
twilio_call_sid = None  # Store Twilio Call SID for recording retrieval

GLOBAL_AUDIO_CAPTURE_QUEUE = []  # 8k PCM16 chunks from call -> to OpenAI

# DTMF pattern matching
DTMF_TAG = re.compile(r"\[\[DTMF:([0-9A-D#*]+)\]\]", re.I)

# ========= Python Logging Setup =========
def setup_logging():
    """Setup Python's logging with timestamped directory"""
    timestamp = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
    call_dir = os.path.join("calls", timestamp)
    os.makedirs(call_dir, exist_ok=True)

    # Setup main application logger
    app_logger = logging.getLogger('jeeves')
    app_logger.setLevel(logging.DEBUG)

    # Main log file handler (everything)
    main_handler = logging.FileHandler(os.path.join(call_dir, "log.txt"), encoding='utf-8')
    main_formatter = logging.Formatter('%(message)s')
    main_handler.setFormatter(main_formatter)
    app_logger.addHandler(main_handler)

    # Conversation transcript handler (clean conversation only)
    conv_logger = logging.getLogger('jeeves.conversation')
    conv_logger.setLevel(logging.INFO)
    conv_handler = logging.FileHandler(os.path.join(call_dir, "call.txt"), encoding='utf-8')
    conv_formatter = logging.Formatter('%(message)s')
    conv_handler.setFormatter(conv_formatter)
    conv_logger.addHandler(conv_handler)

    return call_dir, app_logger, conv_logger

# Global logging setup - will be initialized when call starts
call_dir = None
app_logger = None
conv_logger = None

def log_info(message):
    """Helper function for application logging"""
    if app_logger:
        app_logger.info(message)

def log_conversation(speaker, message):
    """Helper function for conversation logging"""
    line = f"{speaker}: {message}"
    if conv_logger:
        conv_logger.info(line)
    # Also log to main log
    if app_logger:
        app_logger.info(line)

def log_recording_info():
    """Log recording info for manual download later"""
    if twilio_call_sid:
        log_info(f"[RECORDING] Twilio recording enabled for Call SID: {twilio_call_sid}")
        log_info("[RECORDING] Run 'python check_recording.py' to download MP3 when ready")

def wait_for_call_time():
    """Wait until the scheduled call time"""
    if CALL_TIME.upper() == "NOW":
        log_info("[SCHEDULE] Call time is NOW, proceeding immediately")
        return
    
    try:
        from datetime import datetime, timezone
        import zoneinfo
        
        # Parse timezone
        try:
            tz = zoneinfo.ZoneInfo(TIMEZONE)
        except:
            log_info(f"[SCHEDULE] Invalid timezone '{TIMEZONE}', using UTC")
            tz = timezone.utc
        
        # Parse call time - format: DD/MM/YY:HH:MM
        try:
            call_dt = datetime.strptime(CALL_TIME, "%d/%m/%y:%H:%M")
            # Add timezone info
            call_dt = call_dt.replace(tzinfo=tz)
        except ValueError:
            log_info(f"[SCHEDULE] Invalid CALL_TIME format '{CALL_TIME}', expected DD/MM/YY:HH:MM")
            log_info("[SCHEDULE] Proceeding immediately")
            return
        
        # Get current time in the same timezone
        now = datetime.now(tz)
        
        if call_dt <= now:
            log_info(f"[SCHEDULE] Scheduled time {call_dt} is in the past, proceeding immediately")
            return
        
        # Calculate wait time
        wait_seconds = (call_dt - now).total_seconds()
        log_info(f"[SCHEDULE] Waiting until {call_dt} ({TIMEZONE}) - {wait_seconds:.0f} seconds")
        
        # Wait until scheduled time
        while datetime.now(tz) < call_dt:
            remaining = (call_dt - datetime.now(tz)).total_seconds()
            if remaining <= 0:
                break
            
            # Log countdown every minute
            if remaining > 60 and int(remaining) % 60 == 0:
                minutes_left = int(remaining // 60)
                log_info(f"[SCHEDULE] {minutes_left} minutes until call time")
            
            time.sleep(min(10, remaining))  # Check every 10 seconds or remaining time
        
        log_info("[SCHEDULE] Call time reached, proceeding with call")
        
    except Exception as e:
        log_info(f"[SCHEDULE] Error parsing call time: {e}")
        log_info("[SCHEDULE] Proceeding immediately")

# ========= OpenAI Realtime Client =========
class OpenAIRealtimeClient:
    def __init__(self):
        self.api_key = OAI_KEY
        self.ws = None
        self.conversation_active = False
        self.current_transcript = ""
        self.output_port = None  # OpenAIOutputPort (PJSIP source)
        # Watchdog timestamps for mutual silence detection
        self.last_ai_audio_ts = time.monotonic()
        self.last_user_speech_ts = time.monotonic()
        self.closing = False  # Set true when application logic wants to end
        # DTMF sender callback
        self._send_dtmf = None
        # Goodbye handling - wait for audio to finish before ending
        self.goodbye_detected = False
        self.last_audio_chunk_time = time.monotonic()

    async def connect(self):
        url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        headers = [
            ("Authorization", f"Bearer {self.api_key}"),
            ("OpenAI-Beta", "realtime=v1"),
        ]
        log_info("[REALTIME] Connecting to OpenAI...")
        self.ws = await websockets.connect(url, additional_headers=headers)
        await self._send_session_config()
        log_info("[REALTIME] Connected!")

    def attach_output_port(self, port):
        self.output_port = port

    def attach_dtmf_sender(self, sender_callable):
        """Attach DTMF sender callback: lambda digits: ..."""
        self._send_dtmf = sender_callable

    async def _send_session_config(self):
        patient_name = os.getenv("PATIENT_NAME") or "the patient"
        patient_phone = os.getenv("PATIENT_PHONE") or ""
        patient_dob_day = os.getenv("PATIENT_DOB_DAY") or ""
        patient_dob_month = os.getenv("PATIENT_DOB_MONTH") or ""
        patient_dob_year = os.getenv("PATIENT_DOB_YEAR") or ""
        patient_address = os.getenv("PATIENT_ADDRESS") or ""
        appointment_intent = os.getenv("APPOINTMENT_INTENT") or "a general appointment"
        preferred_slots = os.getenv("PREFERRED_SLOTS") or ""

        # Format DOB properly
        patient_dob = ""
        if patient_dob_day and patient_dob_month and patient_dob_year:
            patient_dob = f"{patient_dob_day}/{patient_dob_month}/{patient_dob_year}"

        # Current date/time for appointment context
        current_datetime = datetime.now().strftime("%A, %d %B %Y at %H:%M")
        current_date = datetime.now().strftime("%d/%m/%Y")
        current_day = datetime.now().strftime("%A")

        # Best-turns config: server-side VAD creates responses; we do NOT call response.create after each user turn.
        # We do send ONE response.create for the initial introduction.
        config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "voice": "echo",
                # Keep PCM16 for clarity; we upsample to 24k on send and downsample on receive.
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type": "server_vad",
                    "create_response": True,
                    # Less aggressive VAD for phone audio:
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 800
                },
                "instructions": f"""
You are Jeeves, a polite British personal assistant calling to book medical appointments.

IMPORTANT: This is a live phone call. Keep responses SHORT and natural.

CURRENT DATE & TIME: {current_datetime}
Today is {current_day}, {current_date}

Patient details:
- Name: {patient_name}
- Phone: {patient_phone}
- Date of Birth: {patient_dob}
- Address: {patient_address}
- Reason for visit: {appointment_intent}
- Preferred appointment times: {preferred_slots}

Your task:
1) Introduce yourself: "Good morning, this is Jeeves calling to book an appointment for {patient_name}."
2) Provide the patient details above when asked (DOB, address, phone, etc.).
3) Do not volunteer alternative times; confirm theirs politely or say "I'll need to check with {patient_name}."
4) Keep a professional, courteous British tone.
5) Remember, you are booking an appointment for {patient_name}. You just need the GP to confirm the date and time.
6) You do not assist GP practice, you assist the patient. So do not offer alternative times or assist with booking.
7) Thank them and say goodbye once booking is agreed or they decline.

DTMF/IVR Handling:
- If you hear an automated menu or IVR asking for a key press (e.g., "Press 1 for appointments"):
- DO NOT say the digits aloud in your voice response
- Reply briefly in audio (e.g., "One moment please")
- In the TEXT channel only, output [[DTMF:digits]] exactly, e.g., [[DTMF:1]] or [[DTMF:2#]]
- Then continue the conversation normally after the selection
"""
            },
        }
        await self.ws.send(json.dumps(config))

    async def initial_greeting(self):
        # One-time kick so the assistant says hello; after that, server VAD takes over turn-taking.
        await self.ws.send(json.dumps({"type": "response.create"}))

    async def append_audio24k(self, pcm24_bytes: bytes):
        # Send raw PCM16 (24k) frames as base64, chunked as needed.
        if not pcm24_bytes:
            return
        msg = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm24_bytes).decode()
        }
        await self.ws.send(json.dumps(msg))

    async def listen(self):
        try:
            async for msg in self.ws:
                data = json.loads(msg)
                await self._handle(data)
        except Exception as e:
            log_info(f"[REALTIME] Listen error: {e}")

    async def _handle(self, data: dict):
        t = data.get("type")

        if t == "response.audio.delta":
            # From OpenAI: PCM16 mono @ 24 kHz → downsample to 8 kHz and feed to the port.
            if self.output_port:
                b64 = data.get("delta", "")
                if b64:
                    pcm24 = np.frombuffer(base64.b64decode(b64), dtype=np.int16)
                    pcm8 = _downsample_24k_to_8k(pcm24)
                    self.output_port.enqueue(pcm8.tobytes())
                    self.last_ai_audio_ts = time.monotonic()  # Update AI audio timestamp
                    self.last_audio_chunk_time = time.monotonic()  # Track when audio was last sent

        elif t == "response.audio_transcript.delta":
            self.current_transcript += data.get("delta", "")

        elif t == "response.audio_transcript.done":
            if self.current_transcript:
                # Check for DTMF tokens in the transcript
                text = self.current_transcript
                dtmf_matches = DTMF_TAG.findall(text)
                for digits in dtmf_matches:
                    log_info(f"[DTMF] Requested by AI: {digits}")
                    if self._send_dtmf:
                        self._send_dtmf(digits)

                # Remove DTMF tags from logged conversation
                clean_text = DTMF_TAG.sub("", text).strip()
                if clean_text:
                    log_conversation("JEEVES", clean_text)

                    # Check for goodbye in completed transcript
                    if "goodbye" in clean_text.lower():
                        log_info("[CONVERSATION] Goodbye detected, will end after audio finishes")
                        self.goodbye_detected = True

        elif t == "response.created":
            self.current_transcript = ""

        elif t == "input_audio_buffer.speech_started":
            log_info("[GP]: Started speaking...")
            self.last_user_speech_ts = time.monotonic()  # Update user speech timestamp

        elif t == "input_audio_buffer.speech_stopped":
            log_info("[GP]: Stopped speaking")

        elif t == "input_audio_buffer.committed":
            log_info("[GP]: Audio committed for processing")

        elif t == "conversation.item.input_audio_transcription.completed":
            # This captures what the GP actually said!
            transcript = data.get("transcript", "")
            if transcript:
                log_conversation("GP", transcript)

        elif t == "response.done":
            reason = data.get("response", {}).get("status", "unknown")
            log_info(f"[RESPONSE] Completed with status: {reason}")
            # Goodbye detection moved to transcript.done to ensure proper timing

        elif t == "error":
            error_msg = data.get("error", {}).get("message", "Unknown error")
            log_info(f"[REALTIME] Error: {error_msg}")
            self.conversation_active = False

    async def close(self):
        if self.ws:
            await self.ws.close()
            log_info("[REALTIME] Disconnected")

# ========= PJSUA2 Audio Ports =========
class OpenAIInputPort(pj.AudioMediaPort):
    """Sink: receives decoded 8 kHz PCM16 from the call and buffers ~200 ms chunks."""
    def __init__(self):
        super().__init__()
        self._buf = bytearray()
        self._last_send = 0.0

    def onFrameRequested(self, frame):
        # sink-only; nothing to output
        frame.type = pj.PJMEDIA_FRAME_TYPE_AUDIO

    def onFrameReceived(self, frame):
        try:
            n = frame.size
            if n:
                # frame.buf is a ByteVector; slice to copy
                try:
                    chunk = bytes(frame.buf[:n])
                except Exception:
                    chunk = bytes(frame.buf[i] for i in range(n))
                self._buf.extend(chunk)
                now = time.time()
                if now - self._last_send >= 0.20 and self._buf:
                    GLOBAL_AUDIO_CAPTURE_QUEUE.append(bytes(self._buf))
                    self._buf.clear()
                    self._last_send = now
        except Exception as e:
            log_info(f"[INPUT_PORT] onFrameReceived error: {e}")

class OpenAIOutputPort(pj.AudioMediaPort):
    """Source: serves 8 kHz PCM16 to the call from a FIFO queue."""
    def __init__(self):
        super().__init__()
        self._q = deque()
        self._lock = threading.Lock()

    def enqueue(self, pcm8_bytes: bytes):
        if not pcm8_bytes:
            return
        with self._lock:
            self._q.append(pcm8_bytes)

    def onFrameRequested(self, frame):
        need = frame.size
        out = bytearray()
        with self._lock:
            while need > 0 and self._q:
                chunk = self._q[0]
                take = min(len(chunk), need)
                out += chunk[:take]
                if take == len(chunk):
                    self._q.popleft()
                else:
                    self._q[0] = chunk[take:]
                need -= take
        if need > 0:
            out.extend(b"\x00" * need)  # zero-pad on underrun

        # Ensure exact size match
        if len(out) > frame.size:
            out = out[:frame.size]
        elif len(out) < frame.size:
            out.extend(b"\x00" * (frame.size - len(out)))

        # Try to resize the buffer first (PJSUA2 specific)
        try:
            frame.buf.resize(frame.size)
        except:
            pass  # Some versions might not support resize

        # Copy data with bounds checking
        try:
            # Ensure we don't exceed buffer bounds
            copy_len = min(len(out), frame.size)
            for i in range(copy_len):
                try:
                    frame.buf[i] = out[i]
                except IndexError:
                    log_info(f"[OUTPUT_PORT] Index {i} out of bounds (size={frame.size}), stopping copy")
                    break
        except Exception as e:
            # If all else fails, just provide silence
            log_info(f"[OUTPUT_PORT] Buffer copy failed: {e}, providing silence")
            frame.type = pj.PJMEDIA_FRAME_TYPE_AUDIO
            return

        frame.type = pj.PJMEDIA_FRAME_TYPE_AUDIO

    def onFrameReceived(self, frame):
        # source-only
        pass

# ========= Sample-rate helpers =========
def _upsample_8k_to_24k_bytes(pcm8_bytes: bytes) -> bytes:
    """Repeat each 8 kHz sample 3x to form 24 kHz PCM16. Simple, low-latency."""
    pcm8 = np.frombuffer(pcm8_bytes, dtype=np.int16)
    if pcm8.size == 0:
        return b""
    pcm24 = np.repeat(pcm8, 3)
    return pcm24.tobytes()

def _downsample_24k_to_8k(pcm24: np.ndarray) -> np.ndarray:
    """Box-filter (avg over groups of 3) then decimate to 8 kHz."""
    n = (pcm24.size // 3) * 3
    if n == 0:
        return np.zeros(0, dtype=np.int16)
    t = pcm24[:n].reshape(-1, 3).astype(np.int32)
    avg = (t[:, 0] + t[:, 1] + t[:, 2]) // 3
    avg = np.clip(avg, -32768, 32767).astype(np.int16)
    return avg

# ========= Bridge =========
class AudioBridge:
    def __init__(self, call_media: pj.AudioMedia, realtime_client: OpenAIRealtimeClient):
        self.call_media = call_media
        self.realtime = realtime_client
        self.is_active = False
        self.in_port = None
        self.out_port = None

    async def start(self):
        self.is_active = True
        self.realtime.conversation_active = True

        # (Best effort) register this thread with PJSIP
        try:
            pj.Endpoint.instance().libRegisterThread("AsyncBridge")
        except Exception:
            pass

        # Create 8k/mono/16-bit/20ms ports and wire them to the call
        fmt = pj.MediaFormatAudio()
        fmt.type = pj.PJMEDIA_TYPE_AUDIO
        fmt.clockRate = 8000
        fmt.channelCount = 1
        fmt.bitsPerSample = 16
        fmt.frameTimeUsec = 20000

        self.in_port = OpenAIInputPort();  self.in_port.createPort("oai_in", fmt)
        self.out_port = OpenAIOutputPort(); self.out_port.createPort("oai_out", fmt)

        # Wire: call → in_port ; out_port → call
        self.call_media.startTransmit(self.in_port)
        self.out_port.startTransmit(self.call_media)
        self.realtime.attach_output_port(self.out_port)

        log_info("[BRIDGE] Ports connected. Sending initial greeting...")
        await self.realtime.initial_greeting()

        # Run capture pump + realtime listener
        tasks = [
            asyncio.create_task(self._pump_capture_to_openai()),
            asyncio.create_task(self.realtime.listen()),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        # Log why conversation ended
        for task in done:
            if task.exception():
                log_info(f"[CONVERSATION] Task ended with exception: {task.exception()}")
            else:
                log_info(f"[CONVERSATION] Task completed normally")

        if not self.realtime.conversation_active:
            log_info("[CONVERSATION] Ending: OpenAI conversation inactive")

        self.is_active = False

        # Cleanup
        try: self.call_media.stopTransmit(self.in_port)
        except Exception: pass
        try: self.out_port.stopTransmit(self.call_media)
        except Exception: pass
        self.in_port = None; self.out_port = None
        log_info("[BRIDGE] Stopped")

    async def _pump_capture_to_openai(self):
        log_info("[BRIDGE] Streaming caller audio to OpenAI")
        IDLE_LIMIT = 25.0  # seconds of mutual silence before considering call idle

        while self.is_active and not incoming_call.disconnected.is_set() and self.realtime.conversation_active:
            # Check for mutual silence watchdog
            now = time.monotonic()
            idle_ai = (now - self.realtime.last_ai_audio_ts) > IDLE_LIMIT
            idle_user = (now - self.realtime.last_user_speech_ts) > IDLE_LIMIT

            # Check if goodbye was detected and audio has finished
            if self.realtime.goodbye_detected:
                audio_finished = (now - self.realtime.last_audio_chunk_time) > 2.0  # 2 seconds after last audio
                if audio_finished:
                    log_info("[CONVERSATION] Goodbye audio finished, ending call")
                    self.realtime.conversation_active = False
                    break

            if self.realtime.closing and idle_ai and idle_user:
                log_info("[WATCHDOG] Mutual silence detected, ending call")
                break

            if GLOBAL_AUDIO_CAPTURE_QUEUE:
                pcm8 = GLOBAL_AUDIO_CAPTURE_QUEUE.pop(0)
                pcm24 = _upsample_8k_to_24k_bytes(pcm8)
                await self.realtime.append_audio24k(pcm24)
            else:
                await asyncio.sleep(0.05)

# ========= DTMF Controller =========
class DtmfController:
    def __init__(self, call: pj.Call):
        self.call = call

    def send(self, digits: str, use_rfc2833: bool = True):
        """Send DTMF digits 0-9,*,#,A-D. Sends as RFC2833 by default."""
        try:
            # Preferred: explicit method selection via CallSendDtmfParam
            prm = pj.CallSendDtmfParam()
            prm.digits = digits
            if use_rfc2833 and hasattr(pj, "PJSUA_DTMF_METHOD_RFC2833"):
                prm.method = pj.PJSUA_DTMF_METHOD_RFC2833
            self.call.sendDtmf(prm)
            log_info(f"[DTMF] Sent (RFC2833): {digits}")
        except Exception:
            # Fallback: simple dialDtmf (method is driven by pj stack config)
            try:
                self.call.dialDtmf(digits)
                log_info(f"[DTMF] Sent via dialDtmf: {digits}")
            except Exception as e:
                log_info(f"[DTMF] Failed: {digits} ({e})")

# ========= SIP classes =========
class MyAccount(pj.Account):
    def onRegState(self, prm):
        info = self.getInfo()
        log_info(f"[REG] status={info.regStatus}")
        if info.regStatus >= 400:
            log_info(f"[REG] failed: {info.regStatus}")

    def onIncomingCall(self, prm):
        global incoming_call
        # Logging already initialized in main()
        log_info("[CALL] incoming from Twilio")

        c = MyCall(self, prm.callId)
        op = pj.CallOpParam(); op.statusCode = 200
        c.answer(op)
        incoming_call = c

class MyCall(pj.Call):
    def __init__(self, acc, call_id=pj.PJSUA_INVALID_ID):
        super().__init__(acc, call_id)
        self.am = None
        self.disconnected = threading.Event()

    def onCallState(self, prm):
        ci = self.getInfo()
        log_info(f"[CALL] {ci.stateText} code={ci.lastStatusCode} ({ci.lastReason})")

        if ci.state == pj.PJSIP_INV_STATE_DISCONNECTED:
            log_info("[CALL] hangup after conversation")
            # Log recording info for manual download later
            log_recording_info()
            self.disconnected.set()
            done_event.set()

    def onCallMediaState(self, prm):
        try:
            m = self.getMedia(0)
            self.am = pj.AudioMedia.typecastFromMedia(m)
        except pj.Error:
            return
        adm = pj.Endpoint.instance().audDevManager()
        # Connect both ways initially (we’ll unhook to avoid echo before bridging)
        self.am.startTransmit(adm.getPlaybackDevMedia())
        adm.getCaptureDevMedia().startTransmit(self.am)
        log_info("[MEDIA] bidirectional audio connected")
        media_ready.set()

# ========= Endpoint / registration =========
def build_endpoint():
    ep = pj.Endpoint(); ep.libCreate()
    ep_cfg = pj.EpConfig()

    # Force ALL PJSIP logs to file, NO console output
    ep_cfg.logConfig.consoleLevel = 0  # Completely disable console output
    ep_cfg.logConfig.level = 5  # Full logging level for file

    # Always set up file logging when we have a call directory
    if call_dir:
        pjsip_log_file = os.path.join(call_dir, "pjsip.log")
        ep_cfg.logConfig.filename = pjsip_log_file
        ep_cfg.logConfig.fileFlags = pj.PJ_O_APPEND
        log_info(f"[PJSIP] ALL debug output redirected to: {pjsip_log_file}")
    else:
        # Before call starts - still disable console, minimal file
        ep_cfg.logConfig.consoleLevel = 0
        ep_cfg.logConfig.level = 0  # No logging at all before call starts

    ep_cfg.uaConfig.threadCnt = 0  # manual event pumping
    ep.libInit(ep_cfg)

    ep.transportCreate(pj.PJSIP_TRANSPORT_UDP, pj.TransportConfig())
    ep.libStart()
    log_info("[PJSIP] endpoint started")

    # Prefer G.711 (Twilio)
    try:
        for ci in ep.codecEnum2():
            ep.codecSetPriority(ci.codecId, 0)
        ep.codecSetPriority("PCMU/8000", 255)
        ep.codecSetPriority("PCMA/8000", 254)
    except Exception as e:
        log_info(f"[PJSIP] codec prioritization skipped: {e}")

    return ep

def register_account(ep: pj.Endpoint):
    acc_cfg = pj.AccountConfig()
    acc_cfg.idUri = f"sip:{USER}@{DOMAIN_FQDN}"
    acc_cfg.regConfig.registrarUri = REGISTRAR_URI
    acc_cfg.regConfig.timeoutSec = 900
    acc_cfg.sipConfig.authCreds.append(pj.AuthCredInfo("digest", "*", USER, 0, PASS))

    acc = MyAccount(); acc.create(acc_cfg)

    deadline = time.time() + 20
    while time.time() < deadline:
        ep.libHandleEvents(50)
        info = acc.getInfo()
        if info.regStatus == 200:
            log_info("[REG] Registered OK")
            return acc
        time.sleep(0.02)
    raise RuntimeError("Registration did not reach 200 OK")

# ========= Twilio: ring practice phone, then bridge to SIP =========
def twilio_ring_me_then_bridge():
    client = Client(ACCT_SID, AUTH_TOK)

    # Optional recording webhook (set in .env if you want notifications)
    recording_webhook = os.getenv("RECORDING_WEBHOOK", "")

    # Enhanced TwiML with dual-channel recording
    twiml = f'''<Response>
  <Dial record="record-from-answer-dual"
        recordingTrack="both"
        recordingChannels="dual"
        recordingStatusCallbackEvent="completed"
        recordingStatusCallback="{recording_webhook}">
    <Sip>{AOR_URI}</Sip>
  </Dial>
</Response>'''

    call = client.calls.create(to=TO_NUM, from_=FROM_NUM, twiml=twiml)
    log_info(f"[REST] Calling practice phone {TO_NUM}, will bridge to SIP when answered")
    log_info(f"[REST] Call SID: {call.sid}")
    log_info("[RECORDING] Twilio server-side recording enabled (dual-channel)")

    # Store call SID globally for potential recording retrieval
    global twilio_call_sid
    twilio_call_sid = call.sid

# ========= Realtime conversation =========
async def start_realtime_conversation(call_media, endpoint):
    rt = OpenAIRealtimeClient()
    try:
        await rt.connect()

        # Set up DTMF controller
        dtmf = DtmfController(incoming_call)
        rt.attach_dtmf_sender(lambda digits: dtmf.send(digits, use_rfc2833=True))
        log_info("[DTMF] Controller attached to call")

        bridge = AudioBridge(call_media, rt)
        log_info("[CONVERSATION] Bridge created, starting...")
        await bridge.start()
        log_info("[CONVERSATION] Completed successfully")
    except Exception as e:
        log_info(f"[CONVERSATION] Error: {e}\n{traceback.format_exc()}")
    finally:
        try:
            await rt.close()
        except Exception:
            pass

def run_async_in_thread(coro):
    def runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()
    t = threading.Thread(target=runner)
    t.start()
    return t

# ========= Dispose PJSUA2 objects =========
def dispose_pjsip_objects_before_destroy(ep):
    global incoming_call
    try:
        if incoming_call is not None:
            # Best-effort stop any media pipes
            try:
                adm = pj.Endpoint.instance().audDevManager()
                try: incoming_call.am.stopTransmit(adm.getPlaybackDevMedia())
                except: pass
                try: adm.getCaptureDevMedia().stopTransmit(incoming_call.am)
                except: pass
            except:
                pass
            # Explicitly delete the SWIG wrapper if available
            try:
                if hasattr(incoming_call, "delete"):
                    incoming_call.delete()
            except Exception as e:
                log_info(f"[CLEANUP] Call delete() failed: {e}")
            incoming_call = None
    finally:
        gc.collect()
        time.sleep(0.05)  # give SWIG finalizers a moment

# ========= main =========
def main():

    global call_dir, app_logger, conv_logger

    # Initialize logging FIRST, before PJSIP endpoint
    call_dir, app_logger, conv_logger = setup_logging()

    # Redirect console at OS level (PJSIP is C library, bypasses Python stdout/stderr)
    console_log_file = os.path.join(call_dir, "console.log")

    # Redirect stdout and stderr to file at OS level IMMEDIATELY
    console_fd = os.open(console_log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(console_fd, 1)  # Redirect stdout
    os.dup2(console_fd, 2)  # Redirect stderr
    os.close(console_fd)

    # Prevent child processes from inheriting the redirected file descriptors
    os.set_inheritable(1, False)
    os.set_inheritable(2, False)

    try:
        log_info("[SYSTEM] Logging initialized, console redirected to console.log")

        # Check environment variables after console redirection
        check_environment()

        ep = build_endpoint()
        acc = register_account(ep)  # Keep account alive to prevent garbage collection

        # Wait for scheduled call time before making the call
        wait_for_call_time()

        threading.Thread(target=twilio_ring_me_then_bridge, daemon=True).start()

        # Silent wait - no output needed
        pass  # [WAIT] for incoming call & media...
        deadline = time.time() + 30
        while time.time() < deadline and not (incoming_call and media_ready.is_set()):
            ep.libHandleEvents(50)
        if not (incoming_call and media_ready.is_set()):
            raise RuntimeError("Call/media not established within timeout")

        # short grace period
        t_grace = time.time() + 1.0
        while time.time() < t_grace:
            ep.libHandleEvents(10)

        log_info("[MEDIA] Starting realtime conversation")
        if incoming_call and not incoming_call.disconnected.is_set():
            log_info("[CONVERSATION] Starting OpenAI Realtime API conversation...")

            # Disconnect the initial speaker<->mic pipe to avoid echo; we'll connect via our ports instead.
            try:
                adm = pj.Endpoint.instance().audDevManager()
                incoming_call.am.stopTransmit(adm.getPlaybackDevMedia())
                adm.getCaptureDevMedia().stopTransmit(incoming_call.am)
                log_info("[AUDIO] Disconnected direct audio to prevent echo")
            except Exception as e:
                log_info(f"[AUDIO] Could not disconnect ports: {e}")

            conv_thread = run_async_in_thread(start_realtime_conversation(incoming_call.am, ep))

            # Keep media alive: pump events frequently (threadCnt=0)
            while conv_thread.is_alive() and not incoming_call.disconnected.is_set():
                ep.libHandleEvents(20)
                time.sleep(0.005)

            conv_thread.join(timeout=0.5)

            if incoming_call and not incoming_call.disconnected.is_set():
                log_info("[CALL] hangup after conversation")
                incoming_call.hangup(pj.CallOpParam())
                t_flush = time.time() + 1.0
                while time.time() < t_flush:
                    ep.libHandleEvents(20)

    except Exception as e:
        log_info(f"ERROR: {e}")
    finally:
        try:
            dispose_pjsip_objects_before_destroy(ep)
        except Exception as e:
            log_info(f"[CLEANUP] dispose failed: {e}")

        try:
            ep.libDestroy()
        except Exception:
            pass
        log_info("[PJSIP] endpoint destroyed")

if __name__ == "__main__":
    main()
