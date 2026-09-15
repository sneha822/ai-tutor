# Setup guide

This guide takes you from nothing to talking with the AI tutor. It takes about 20–30 minutes, and most of that is
waiting for downloads. You only do it once.

> **Tested on Mac** (Apple Silicon). Windows and Linux steps are included and should work, but haven't been tested
> yet. If something goes wrong, check [Problems and fixes](#problems-and-fixes) at the bottom.

## What you need

- A laptop or desktop with a **webcam** and a **microphone**
- **Headphones or earphones** (strongly recommended: through speakers, the tutor can hear itself and stop talking)
- About **5 GB of free disk space**
- **Internet** during setup (afterwards, only the tutor's answers need internet)
- A free **Groq account**, for the API key (step 1)

## Step 1: Get your free Groq API key

The tutor's "brain" runs on Groq. The API key is like a password that lets the app use it.

1. Go to <https://console.groq.com> and sign up (Google login works).
2. Open **API Keys** and click **Create API Key**. Name it anything, like `ai-tutor`.
3. **Copy the key** (it starts with `gsk_`) and paste it somewhere safe for step 5. Groq shows it only once.

Never share this key or post it online.

## Step 2: Install the tools

Follow the part for your computer.

### Mac

1. Open **Terminal**: press `Cmd + Space`, type `Terminal`, press Enter.
2. If you don't have Homebrew yet, go to <https://brew.sh>, copy the install command shown on that page, paste it into
   Terminal and press Enter. When it finishes, it shows a few **Next steps** commands: run those too.
3. Install Python 3.12, a voice helper (eSpeak NG) and git:

   ```bash
   brew install python@3.12 espeak-ng git
   ```

### Windows

1. **Python 3.12:** from <https://www.python.org/downloads/windows/>, download the **Windows installer (64-bit)** for
   a **3.12** version (3.13 and newer won't work). On the first screen of the installer, **tick "Add python.exe to
   PATH"**, then click Install Now.
2. **Git:** install from <https://git-scm.com/download/win>. The default options are fine.
3. **eSpeak NG** (helps the tutor pronounce unusual words): open
   <https://github.com/espeak-ng/espeak-ng/releases>, download the `.msi` file (the `x64` one for most PCs) and
   install it.
4. Open **PowerShell**: Start menu, type `PowerShell`, press Enter. You'll type the commands below in there.

### Linux (Ubuntu or Debian)

```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv espeak-ng libportaudio2 git
```

If `python3.12` can't be found, your system is older than Ubuntu 24.04; install Python 3.12 with
[pyenv](https://github.com/pyenv/pyenv) first.

## Step 3: Download the project

Download it with git:

```bash
git clone https://github.com/sneha822/ai-tutor.git
```

Then go into the folder:

```bash
cd ai-tutor
```

**Every command after this runs inside the `ai-tutor` folder.** If you close the terminal, open a new one and run
`cd ai-tutor` again first.

## Step 4: Install the Python packages

This creates a private Python folder for the project (called `.venv`) and installs everything into it. It downloads
about 2 GB, so expect 5–15 minutes.

**Mac:**

```bash
"$(brew --prefix)/bin/python3.12" -m venv .venv
```

```bash
.venv/bin/python -m pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
```

```powershell
.venv\Scripts\python -m pip install -r requirements.txt
```

**Linux:**

```bash
python3.12 -m venv .venv
```

```bash
.venv/bin/python -m pip install -r requirements.txt
```

It's finished when the last line starts with `Successfully installed`. Yellow warnings are normal. A red `ERROR` is
not: see [Problems and fixes](#problems-and-fixes).

## Step 5: Add your API key

1. Make your own copy of the settings file:
   - **Mac / Linux:** `cp .env.example .env`
   - **Windows:** `copy .env.example .env`
2. Open it:
   - **Mac:** `open -e .env`
   - **Windows:** `notepad .env`
   - **Linux:** `nano .env`
3. Replace `your_groq_api_key_here` with your key from step 1, so the line looks like this (with your real key):

   ```
   GROQ_API_KEY=gsk_abc123yourkeyhere
   ```

   No spaces and no quotes.
4. Save and close. (In `nano`: press `Ctrl+O`, Enter, then `Ctrl+X`.)

The `.env` file stays on your computer. The project is set up so git never uploads it.

## Step 6: Download the AI models

This downloads the tutor's voice, speech recognition and face tracking (about 600 MB). They run on your own computer.

- **Mac / Linux:** `.venv/bin/python scripts/download_models.py`
- **Windows:** `.venv\Scripts\python scripts\download_models.py`

Wait for `All models cached.` at the end. Every line above it should start with `[OK]`.

## Step 7: Start the tutor

**Mac.** Use the launcher, so macOS lets Terminal use your camera and microphone:

```bash
open scripts/run_tutor.command
```

A new Terminal window opens. The first time, macOS asks whether Terminal may use the **Camera** and the
**Microphone**: click **Allow** both times.

**Windows:**

```powershell
.venv\Scripts\python main.py
```

**Linux:**

```bash
.venv/bin/python main.py
```

Starting takes up to a minute. **Look at your screen while it starts:** it uses the first 2 seconds to learn what
"paying attention" looks like for you. When the terminal says `READY`, your browser opens the tutor page. If it
doesn't, go to <http://127.0.0.1:8765> yourself.

The browser will ask for camera access too. That's only for the small video of yourself in the corner, which stays in
your browser.

## Step 8: Tell it who you are

A welcome window appears on the page:

1. Type your **name**.
2. Under **I'd like to**, pick a subject to study (or **Another subject…** and type it), or pick an off-topic mode:
   **Talk with a friend**, **Therapist** (a supportive listener, not a real therapist) or **Just a normal chat**.
3. Choose whether to **use your camera**. The tutor needs it to notice when you get distracted. The other modes work
   fine without it, but they're a little warmer when they can see you.
4. Click **Start**. It greets you by name and tells you what you're doing today.

To switch later, click your name at the top right of the page.

## Step 9: Choose your microphone and speaker

On the tutor page, click the **small arrow (^) just left of the microphone button** at the bottom.

- **Microphone:** click the one you're using. Say something: the green bar next to "Microphone" should move.
- **Speaker:** click your headphones, then press **Test**. You should hear a short chime.

You can mix and match, for example headphones for sound with your laptop's microphone. Your choice is remembered for
next time.

## Step 10: Use it

Just talk. Try *"Teach me derivatives"*, or in friend mode *"Guess what happened today"*.

- **Interrupt anytime** by talking over it.
- **Watch its face.** Its colour and expression change with how it feels: happy, sad, confused, surprised, angry
  and more. The chip at the top says which.
- **In tutor mode it notices when you drift.** Look at your phone or step away for about 10 seconds and it brings
  you back. Keep doing it and it gets annoyed, then properly mad (never mean).
- **The bottom bar:** microphone on/off (with the device menu next to it), camera on/off, stop the tutor talking,
  trigger a focus reminder now, pause focus reminders, recalibrate (look at the screen, then click), and the shield,
  which shows exactly what is sent to the AI and what stays on your computer.
- **Your camera tile:** drag it to any corner, resize it from its inner corner, or minimize it.
- **To quit:** click on the terminal window, type `q` and press Enter.

### Teaching from your own notes

Put `.md`, `.txt` or `.pdf` files in the `materials` folder and restart the tutor. It will teach from them. A sample
calculus file is included; delete it if you don't need it.

### Starting it again another day

- **Mac:** in Finder, open the `ai-tutor` folder, then `scripts`, and double-click `run_tutor.command`.
- **Windows:** open PowerShell, run `cd ai-tutor`, then `.venv\Scripts\python main.py`.
- **Linux:** run `cd ai-tutor`, then `.venv/bin/python main.py`.

### Getting updates

When the project gets updated on GitHub, run this inside the `ai-tutor` folder:

```bash
git pull
```

Then run the step 4 install command and the step 6 download command again. They only fetch what changed.

## Problems and fixes

| Problem | Fix |
|---|---|
| `python3.12: command not found`, or `py -3.12` doesn't work | Python 3.12 isn't installed (step 2). On Windows, run the installer again and tick **Add python.exe to PATH**. |
| pip says no matching version was found for a package | You're using the wrong Python version. Delete the `.venv` folder and redo step 4 with Python 3.12. |
| An error about `GROQ_API_KEY`, or the tutor says "I lost my connection" | Check step 5: the file must be called exactly `.env` (not `.env.txt`) and hold your real key with no spaces. Also check your internet connection. |
| The tutor doesn't react when you talk | Make sure you clicked **Start** in the welcome window (step 8). Then open the device menu (step 9) and pick another microphone until the green bar moves. Mac: System Settings → Privacy & Security → Microphone → turn on **Terminal**, then restart the tutor. Windows: Settings → Privacy & security → Microphone → turn on **Let desktop apps access your microphone**. |
| Focus card says "Camera off" when the camera is on | Mac: System Settings → Privacy & Security → Camera → turn on **Terminal**, then restart. Windows: Settings → Privacy & security → Camera → turn on **Let desktop apps access your camera**. Close other apps using the camera (Zoom, Meet, Teams). |
| The tutor keeps stopping mid-sentence | It hears its own voice through your speakers. Use headphones and pick them as the speaker in the device menu. |
| No sound | Device menu → Speaker → pick the right one and press **Test**. Check your volume. |
| Windows: the small camera preview says "unavailable" | Windows lets only one program use the webcam at a time, so the app shows its own camera view in that tile. Run `git pull`, restart the tutor and reload the page. Close Zoom, Teams or the Camera app if they have the camera. |
| The voice starts late, or focus tracking feels slow | Plug the laptop in (battery saver slows the CPU a lot), close other heavy apps and browser tabs, and use headphones. The app already limits its own CPU use and lowers the 3D quality on slow computers. Still slow? Share `logs/tutor.log`: its `LATENCY` and `infer=` numbers show which part is slow. |
| The page says "offline" or won't load | Wait for `READY` in the terminal, then reload <http://127.0.0.1:8765>. If the terminal says the port is in use, a tutor is already running: quit it with `q`, or restart the computer. |
| Windows: `DLL load failed` when starting | Install the Microsoft Visual C++ Redistributable from <https://aka.ms/vs/17/release/vc_redist.x64.exe> and try again. |
| Keyboard shortcuts (like Ctrl+Option+I) do nothing | Mac: System Settings → Privacy & Security → **Input Monitoring** → turn on Terminal. You can also just use the buttons on the page. |
| Step 6 says `Some downloads FAILED` | Usually the internet dropped for a moment. Run the step 6 command again. |

Still stuck? Share the last 30 lines of `logs/tutor.log` (in the project folder) with whoever sent you this project.
That file includes what you said to the tutor, so read it before sending.
