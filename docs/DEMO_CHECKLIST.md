# DEMO_CHECKLIST

Project: `~/Downloads/science_teacher_demo_pack`

## 1. Startup

Assumption:
- `DASHSCOPE_API_KEY` is already available in the shell environment
- The current OpenAI-compatible proxy LLM key is already available in the shell environment

Start the demo:

```bash
cd ~/Downloads/science_teacher_demo_pack
source .venv/bin/activate
RECORDER_MAX_SECONDS=5 SCIENCE_TEACHER_LOG_LEVEL=DEBUG WAKEWORD_ENABLED=1 WAKEWORD_THRESHOLD=0.2 uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open:
- `http://127.0.0.1:8000`

Expected startup log signals:
- `startup configuration: ... providers_configured=True`
- `wake word listener thread started`
- `local audio playback enabled=True`

## 2. Primary Demo Flow

What to say:
1. Say `Alexa`
2. Wait for the acknowledgement `我在，请讲`
3. Ask a short science question

Recommended test questions:
- `雨是怎么形成的？`
- `为什么白天看得到太阳？`
- `彩虹为什么会出现？`

Expected visible state transitions:
1. `idle`
2. `listening`
3. `thinking`
4. `answering`
5. `idle`

Expected visible outputs:
- The state badge changes color and label at each stage
- The recognized question appears in the question panel
- The answer appears in the answer panel with readable paragraphs and numbered points
- The avatar mouth moves while acknowledgement or answer audio is playing
- The answer audio plays from the local machine

Expected generated artifacts:
- `latest/question.wav`
- `latest/answer.json`
- `latest/answer.mp3`

## 3. Manual Fallback Trigger

If wake word is unreliable in the room:
1. Open the page
2. Click `麦克风提问`
3. Ask the question normally

Expected behavior:
- The same real pipeline still runs
- The same question and answer panels update
- Local audio playback still starts after TTS

## 4. Logs To Watch

The most useful backend log lines are:
- `wake detected: ...`
- `recording started: ...`
- `recording ended: ...`
- `ASR completed: ...`
- `agent completed: ...`
- `TTS completed: ...`
- `playback started: ...`
- `playback ended: ...`

If the UI and audio disagree, trust the terminal logs first.

## 5. Common Recovery Steps

Wake word not triggering:
- Confirm the mic is correct in startup logs
- Speak `Alexa`, then pause before the question
- Retry with a quieter room
- Lower `WAKEWORD_THRESHOLD` to `0.1` or `0.15` if needed

It records but does not understand the question:
- Check `DASHSCOPE_API_KEY`
- Watch for `ASR completed`
- Verify `latest/question.wav` was updated

It understands but does not answer:
- Check the current OpenAI-compatible proxy LLM key
- Watch for `agent completed`
- If you see `401`, rotate or re-export the LLM key

Text appears but no sound:
- Check `playback started` and `playback ended`
- Confirm macOS output device is correct
- Confirm `LOCAL_AUDIO_PLAYBACK_ENABLED` was not disabled
- If needed, open `latest/answer.mp3` manually to verify the file exists

UI looks stuck:
- Refresh `http://127.0.0.1:8000`
- Confirm `ws open` appears in browser console if debugging the page
- Use the manual button once to verify the backend path is still healthy

## 6. Presenter Notes

For the smoothest live demo:
- Use a short question
- Wait for the acknowledgement before speaking the question
- Keep the terminal visible on a second screen if you want quick debugging
- Keep the browser full-screen on the demo display

Current temporary limitation before Live2D:
- The avatar is still the temporary face component
- Mouth motion is improved, but it is not yet a production avatar renderer
