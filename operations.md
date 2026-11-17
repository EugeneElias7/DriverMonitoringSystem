# DMS Operations Reference

## Pi (Raspberry Pi)

### Start (headless, background)
```bash
ssh sansolution@192.168.1.14 "bash /home/sansolution/dms/start_pi.sh"
```

### Start (headless, foreground — see detections in terminal)
```bash
ssh sansolution@192.168.1.14
cd /home/sansolution/dms
.venv/bin/python run.py --config config_pi.json --source picamera --headless
```
Press `Ctrl+C` to stop.

### Start (with preview window — needs monitor/VNC)
```bash
cd /home/sansolution/dms
.venv/bin/python run.py --config config_pi.json --source picamera
```
Press `q` in the window to quit.

### Record preview to file (no display needed)
```bash
cd /home/sansolution/dms
.venv/bin/python run.py --config config_pi.json --source picamera --headless --record /home/sansolution/dms/preview.mp4 --max-seconds 60
```

### Watch live log
```bash
ssh sansolution@192.168.1.14 "tail -f /home/sansolution/dms/logs/app.log"
```

### Kill running instance
```bash
ssh sansolution@192.168.1.14 "pkill -f '[r]un\.py'"
```

### Check temperature and throttle status
```bash
ssh sansolution@192.168.1.14 "cat /sys/class/thermal/thermal_zone0/temp; vcgencmd get_throttled"
```
- Temperature in millidegrees (divide by 1000 for °C). Normal: 60-75 °C under load.
- `throttled=0x0` = no throttling. `0x70000` = past events only. `0x70005` = current undervoltage.

### Check running process
```bash
ssh sansolution@192.168.1.14 "ps -o pid,etime,pcpu,pmem,cmd -p \$(pgrep -f 'run.py')"
```

### Run tests on Pi
```bash
ssh sansolution@192.168.1.14 "cd /home/sansolution/dms && .venv/bin/python -m unittest discover -s tests -q"
```

### Deploy from PC to Pi
```bash
cd "C:\D drive\DMS-Model-Testing"
bash deploy/upload.sh
```

### Check Pi IP
```bash
ssh sansolution@192.168.1.14 "hostname -I"
```

---

## PC (Windows)

### Run with webcam (preview)
```bash
cd "C:\D drive\DMS-Model-Testing"
python run.py --config config_pc.json --source 0
```

### Run with webcam (headless, terminal output)
```bash
cd "C:\D drive\DMS-Model-Testing"
python run.py --config config_pc.json --source 0 --headless
```

### Run with a video file
```bash
cd "C:\D drive\DMS-Model-Testing"
python run.py --config config_pc.json --source path\to\clip.mp4 --loop --realtime
```

### Run tests on PC
```bash
cd "C:\D drive\DMS-Model-Testing"
python -m pytest tests -q
```

### Generate audio WAV files (Neeraja/Indian voice)
```powershell
# Install Neeraja voice first: Windows Settings -> Time & Language -> Speech -> Add voices -> English (India)
# Then run:
cd "C:\D drive\DMS-Model-Testing"
python -c "
import pyttsx3
engine = pyttsx3.init()
# Select Indian voice
for v in engine.getProperty('voices'):
    if 'Neerja' in v.name or 'Heera' in v.name or 'Ravi' in v.name or 'India' in v.name:
        engine.setProperty('voice', v.id)
        print('Using:', v.name)
        break
phrases = {
    'audio/stop_smoking.wav': 'Stop smoking',
    'audio/do_not_use_phone.wav': 'Do not use phone',
    'audio/please_take_a_break.wav': 'Please take a break',
    'audio/system_on.wav': 'Driver monitoring active',
}
for path, text in phrases.items():
    engine.save_to_file(text, path)
    print('Saved', path)
engine.runAndWait()
"
```

### Check git status
```bash
cd "C:\D drive\DMS-Model-Testing"
git status --short
git log --oneline -5
```

---

## SSH / Network

### SSH into Pi
```bash
ssh sansolution@192.168.1.14
```

### Copy a file from PC to Pi
```bash
scp file.txt sansolution@192.168.1.14:/home/sansolution/dms/
```

### Copy a file from Pi to PC
```bash
scp sansolution@192.168.1.14:/home/sansolution/dms/preview.mp4 "C:\D drive\DMS-Model-Testing\"
```

---

## Troubleshooting

### App won't start / port in use
```bash
ssh sansolution@192.168.1.14 "pkill -9 -f 'run.py'; sleep 2; bash /home/sansolution/dms/start_pi.sh"
```

### No face detected
- Check camera connection: `ssh sansolution@192.168.1.14 "libcamera-hello --timeout 2000"`
- Check lighting and camera angle
- Verify face analyzer model exists: `ls /home/sansolution/dms/models/face_landmarks.onnx`

### Detectors not running
- Check `detectors_paused` in status line — if `yes`, face is missing or eyes are closed
- Check detector models exist: `ls /home/sansolution/dms/models/*.onnx`

### Audio not playing
- Test speaker: `ssh sansolution@192.168.1.14 "speaker-test -t wav -c 2 -l 1"`
- Test aplay: `ssh sansolution@192.168.1.14 "aplay -D plughw:Headphones /home/sansolution/dms/audio/system_on.wav"`
- Check audio device in config_pi.json: `"device": "plughw:Headphones"`

### IR LED control
The IR LEDs are controlled via GPIO pin 18 (configurable in `camera.ir_gpio_pin` in config_pi.json). They turn on automatically when frame brightness drops below 80% (configurable via `camera.ir_brightness_threshold`).

```bash
# Check IR status in the live log
ssh sansolution@192.168.1.14 "tail -f /home/sansolution/dms/logs/app.log | grep ir="
```

### Test IR manually (Python on Pi)
```bash
ssh sansolution@192.168.1.14 "cd /home/sansolution/dms && .venv/bin/python -c '
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)
GPIO.output(18, GPIO.HIGH)
print(\"IR ON\")
import time; time.sleep(2)
GPIO.output(18, GPIO.LOW)
GPIO.cleanup(18)
print(\"IR OFF\")
'"
```


### False detections
- Check status line: `ear=`, `mar=`, `gap=`, `gapH=`, `det=`
- Adjust thresholds in config_pi.json
- Run tests after changes: `.venv/bin/python -m unittest discover -s tests -q`
```

Now deploy the audio.py changes and the new WAV files.<|end|><|start|>start to=shell<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name="command">
& 'C:\Program Files\Git\bin\bash.exe' deploy/upload.sh