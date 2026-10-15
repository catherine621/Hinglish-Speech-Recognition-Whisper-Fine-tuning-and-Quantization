# Code-Switched Hinglish ASR: LoRA fine-tuning of openai/whisper-small with edge deployment

## Problem
Indian speakers mix Hindi and English within a sentence. General ASR models degrade on such speech.

## Data
Public corpus `addyo07/noisy-hinglish-asr` (Apache-2.0). Kept clean `hinglish` clips only; train 3000 clips
(3.21 h), val 161, test 229. Test sentences overlapping train were removed.
Transcripts are lowercased and punctuation-stripped for training and scoring.

## Method
LoRA (r=32) on attention projections, 800 steps, effective batch 16, speed + noise augmentation.
LoRA merged into the base model, then converted to CTranslate2 INT8 and (optionally) whisper.cpp ggml with q8_0/q5_1/q4_0.

## Results (held-out test)
|                                        |    WER |    CER |   EngWordRecall |
|:---------------------------------------|-------:|-------:|----------------:|
| Baseline: forced Hindi, no guard       | 196.07 | 130.14 |            0    |
| Baseline: auto language, no guard      |  96.63 |  65.51 |           58.19 |
| Baseline: auto language + no-repeat-6  |  71.11 |  52.6  |           58.19 |
| Baseline: auto language + no-repeat-10 |  79.45 |  55.08 |           58.19 |
| Fine-tuned (LoRA, merged)              |  16.88 |   8.26 |           91.85 |

Edge benchmark (emulated: 2 CPU threads, INT8):
|                           |   Size_MB |    WER |   CER |   EngWordRecall |    RTF |   PeakRAM_MB |   Load_s |
|:--------------------------|----------:|-------:|------:|----------------:|-------:|-------------:|---------:|
| Baseline FP16 (HF, GPU)   |    487.47 |  71.11 | 52.6  |           58.19 | nan    |       nan    |   nan    |
| Fine-tuned FP16 (HF, GPU) |    970.93 |  16.88 |  8.26 |           91.85 | nan    |       nan    |   nan    |
| Fine-tuned INT8 (CT2)     |    253.21 |  22.5  | 15.46 |           89.77 |   1.26 |       653.84 |     0.59 |
| Baseline INT8 (CT2)       |    249.61 | 111.83 | 84.78 |            0    |   1.51 |       655.29 |     0.59 |

## Limitations
- Dataset has no speaker IDs, so unseen-speaker evaluation is not guaranteed.
- Edge numbers are from a Colab CPU, not a physical phone / Raspberry Pi (add phone numbers if you test on a device).
- Small training subset; results will vary with seed and subset size.
