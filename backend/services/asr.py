try:
    from .asr_aliyun import ASRError, QUESTION_AUDIO, transcribe_question
except ImportError:
    from asr_aliyun import ASRError, QUESTION_AUDIO, transcribe_question
