# Jeeves - AI GP Appointment Booking System

**Standalone script using Twilio SIP + OpenAI Realtime API for natural phone conversations.**

## 🎯 What It Does

Jeeves makes automated calls to GP practices to book appointments using natural conversation AI. The system connects via Twilio SIP, uses OpenAI's Realtime API for speech-to-speech conversation, and saves complete call logs and recordings.

**Key Features:**
- 🤖 **Natural conversation** powered by OpenAI Realtime API (speech-to-speech)
- 📞 **Twilio SIP integration** for real phone calls
- 🎙️ **Server-side recording** with dual-channel audio (caller + practice)
- 📁 **Local call logs** with timestamped folders
- ⏰ **Call scheduling** with timezone support
- 📋 **Clean transcripts** and conversation logs
- ⚙️ **Simple configuration** via `.env` file

## 🚀 Quick Start

### Prerequisites

**Python Version:** This project requires **Python 3.13.4** (specified in `.python-version`)

**Why Python 3.13.4?**
- Latest Python with performance improvements
- Compatible with all dependencies (pjsua2, OpenAI, etc.)
- Tested and working with the current package versions

### 1. Installation

```bash
# Clone or navigate to the project directory
cd jeeves

# Option A: Using pyenv (recommended)
pyenv install 3.13.4
# pyenv will automatically use this version due to .python-version file

# Option B: Without pyenv
# Make sure you have Python 3.13.4 installed system-wide

# Verify Python version
python --version  # Should show: Python 3.13.4

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Don't have pyenv?** Install it first:
```bash
# macOS
brew install pyenv

# Linux
curl https://pyenv.run | bash
```

### 2. Set up Twilio SIP

You need a Twilio account with:
- **Phone number** for outbound calls
- **SIP Domain** configured
- **SIP credentials** (username/password)

### 3. Configure Everything in `.env`

Copy `.env.example` to `.env` and fill in your details.


### 4. Run Jeeves

```bash
# Make an appointment booking call
python gpt-5-test.py
```

## 📁 Output Structure

Each call creates a timestamped folder in `calls/`:

```
calls/
└── 22-09-25_18-11-27/        # DD-MM-YY_HH-MM-SS format
    ├── call.txt              # Clean conversation transcript
    ├── log.txt               # System logs and events
    ├── console.log           # PJSIP debug output
    └── pjsip.log             # Detailed SIP protocol logs
```

### Sample `call.txt`:

```
JEEVES: Good morning, this is Jeeves calling to book an appointment for John Doe.
GP: Hi, what can I help you with?
JEEVES: I'd like to book an appointment for John Doe. His date of birth is 15th March 1985.
GP: What's the appointment for?
JEEVES: It's for a routine checkup.
GP: I have availability next Tuesday at 10:30 AM, would that work?
JEEVES: That sounds perfect. Please book that appointment.
GP: All done, the reference is GP12345.
JEEVES: Thank you very much. Goodbye.
```

### Sample `log.txt`:

```
[SYSTEM] Logging initialized, console redirected to console.log
[SCHEDULE] Call time is NOW, proceeding immediately
[REST] Calling practice phone +447552761633, will bridge to SIP when answered
[RECORDING] Twilio server-side recording enabled (dual-channel)
[CALL] incoming from Twilio
[CONVERSATION] Starting OpenAI Realtime API conversation...
JEEVES: Good morning, this is Jeeves calling to book an appointment for John Doe.
GP: Hi, what can I help you with?
[CONVERSATION] Goodbye detected, will end after audio finishes
[CONVERSATION] Goodbye audio finished, ending call
[RECORDING] Twilio recording enabled for Call SID: CA1234567890abcdef
```

## 🎙️ Natural Conversation Flow

The AI handles everything naturally:

```
🎧 "Hello, City Medical Practice"
🤖 "Hi, I'd like to book an appointment for John Doe"

🎧 "What's the appointment for?"
🤖 "It's for a routine checkup"

🎧 "What's his date of birth?"
🤖 "15th of March, 1985"

🎧 "I have Monday at 10:30 AM, would that work?"
🤖 "Perfect, that's in my preferred morning slot. Please book it."
```

## 🎙️ Call Recordings

The system automatically enables **Twilio server-side recording** with dual-channel audio:
- **Channel 1**: Jeeves (AI assistant)
- **Channel 2**: GP practice receptionist

The recordings can be accessed in Twilio account, Monitor section > Call recordings.

### Download Recordings

Use the included script to locally download MP3 recordings:

```bash
# Download all recent recordings
python get-recordings.py
```

**Output:**
```
Found 2 recordings in the last hour:

--- Recording 1 ---
SID: RE1e8a43d851650298ab8bb94e835e3bc4
Call SID: CAf1a22db1b73ef775fa4728cedbdd376a
Duration: 39 seconds
Status: completed
Channels: 2
✅ Downloaded: recordings/recording_RE1e8a43d851650298ab8bb94e835e3bc4.mp3
```

**Features:**
- ✅ **Auto-creates** `recordings/` directory
- ✅ **Skips duplicates** - won't re-download existing files
- ✅ **Proper error handling** with clear status messages
- ✅ **Timezone-aware** datetime handling

## 📋 Configuration Format

**Patient DOB:** Separate day/month/year to avoid date format confusion
- `PATIENT_DOB_DAY=15`
- `PATIENT_DOB_MONTH=3`
- `PATIENT_DOB_YEAR=1985`

**Call Time:** Schedule calls or run immediately
- `CALL_TIME=NOW` - Run immediately
- `CALL_TIME=23/09/25:14:30` - Schedule for specific date/time
- `TIMEZONE=Europe/London` - Timezone for scheduled calls

**Preferred Slots:** Appointment preferences
- Format: `DD/MM/YY:HHMM-HHMM`
- Examples:
  - `30/09/25:0900-1200` - Sept 30, 2025, 9 AM to 12 PM
  - `1/10/25:1400-1700` - Oct 1, 2025, 2-5 PM

## 🏗️ System Architecture

```
Configuration (.env) → gpt-5-test.py → Call Logs (calls/folder/)
                            ↓
                    Twilio SIP + PJSIP
                            ↓
                  OpenAI Realtime API
                            ↓
                    Server-side Recording
```

**Core Files:**
- `gpt-5-test.py` - Main application script
- `get-recordings.py` - Download call recordings
- `.env` - Configuration (copy from `.env.example`)
- `.gitignore` - Excludes logs and recordings from git

**Key Components:**
- **PJSIP** - SIP protocol handling
- **Twilio** - Phone network + recording
- **OpenAI Realtime API** - Speech-to-speech conversation
- **Local logging** - All conversations saved locally

## 🎯 Key Benefits

- ✅ **Real phone calls** - works with actual GP practices
- ✅ **Natural conversation** - speech-to-speech AI
- ✅ **Complete recordings** - dual-channel MP3 files
- ✅ **No database** - results saved as files
- ✅ **Call scheduling** - run immediately or at specific times
- ✅ **Clean transcripts** - only dialogue, no system messages
- ✅ **Organized results** - timestamped folders
- ✅ **Silent operation** - all output goes to log files

## 🚀 Usage

### Make a Call

```bash
# Run immediately
python gpt-5-test.py
```

### Check Results

```bash
# View call logs
ls calls/
cat calls/22-09-25_18-11-27/call.txt

# Download recordings
python get-recordings.py
ls recordings/
```

## 🛠️ Troubleshooting

### Common Issues

**Problem: Call fails to connect**
- Check Twilio SIP credentials are correct
- Verify SIP domain is properly configured in Twilio
- Ensure phone numbers have correct country codes

**Problem: No audio/conversation doesn't work**
- Check OpenAI API key is valid
- Verify `pjsua2` is properly installed
- Look at `pjsip.log` for detailed SIP debugging

**Problem: Recording not available**
- Recordings take 1-2 minutes to process after call ends
- Check the Call SID in logs matches the recording
- Use `get-recordings.py` to download manually

**Problem: Environment variables not loading**
- Ensure `.env` file exists (copy from `.env.example`)
- No quotes needed around values in `.env`
- Check file is in the same directory as the script

### Installation Issues

**Problem: `pjsua2` installation fails**
```bash
# macOS - install via Homebrew first
brew install pjproject
pip install -r requirements.txt

# Linux - install development headers
sudo apt-get install libpjproject-dev
pip install -r requirements.txt
```

**Problem: Permission errors on macOS**
```bash
# Accept Xcode license if prompted
sudo xcodebuild -license agree
```

---

**Jeeves** - Real AI phone calls for appointment booking! 🤖📞
